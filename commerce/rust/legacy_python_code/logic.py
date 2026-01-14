import os
import re
import json
import uuid
import sys
import gc
import lancedb
import pyarrow as pa
import numpy as np
import torch
import torch.nn.functional as F
import threading
import time
import psutil
from PIL import Image
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Model Configuration
GEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct" 
EMBED_MODEL_ID = "google/embeddinggemma-300m"
HF_TOKEN = "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"

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOCAL_THINKING_PATH = os.path.join(CACHE_DIR, "Qwen3-VL-2B-Thinking")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

class TradeLogisSystem:
  def __init__(self, data_dir: str = DATA_DIR):
    self.db = lancedb.connect(data_dir)
    self.table_name = "trade_documents"
    self.compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = None
    self.processor = None
    self.embed_tokenizer = None
    self.embed_model = None
    self._load_embedding_model()

  def _load_embedding_model(self):
    try:
      local_embed_path = os.path.join(CACHE_DIR, "embeddinggemma-300m")
      load_path = local_embed_path if os.path.exists(os.path.join(local_embed_path, "config.json")) else EMBED_MODEL_ID
      self.embed_tokenizer = AutoTokenizer.from_pretrained(load_path, token=HF_TOKEN, cache_dir=CACHE_DIR)
      self.embed_model = AutoModel.from_pretrained(load_path, token=HF_TOKEN, device_map="cpu", dtype=torch.float32, cache_dir=CACHE_DIR)
      self.embed_model.eval()
    except: pass

  def _get_table(self):
    try:
      return self.db.open_table(self.table_name)
    except:
      return None

  def _get_embedding(self, text: str) -> List[float]:
    if self.embed_model is None: return [0.0] * 1024
    with torch.inference_mode():
      inputs = self.embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cpu")
      outputs = self.embed_model(**inputs)
      mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
      emb = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
      return F.normalize(emb, p=2, dim=1)[0].cpu().float().numpy().tolist()

  def set_device_mode(self, mode: str):
    if self.model: self.unload_model()
    self.device = "cuda" if mode in ["Auto", "GPU"] and torch.cuda.is_available() else "cpu"
    return f"Device set to {self.device.upper()}."

  def _load_main_model(self):
    if self.model: return
    if self.device == "cpu":
      try:
        phys_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
        safe_threads = max(1, phys_cores - 2)
        torch.set_num_threads(safe_threads)
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        safe_mem_gb = total_gb - 4.5 if total_gb > 8 else total_gb * 0.5
        max_mem = {"cpu": f"{int(safe_mem_gb)}GiB"}
      except: max_mem = None
      device_map = "cpu"; quant_config = None
    else:
      bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self.compute_dtype, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
      device_map = "auto"; max_mem = {0: "4GiB", "cpu": "16GiB"}; quant_config = bnb_config
    try:
      self.model = AutoModelForImageTextToText.from_pretrained(GEN_MODEL_ID, quantization_config=quant_config, device_map=device_map, max_memory=max_mem, trust_remote_code=True, cache_dir=CACHE_DIR, dtype=self.compute_dtype, low_cpu_mem_usage=True, attn_implementation="sdpa")
      self.processor = AutoProcessor.from_pretrained(GEN_MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
    except: pass

  def unload_model(self):
    if self.model:
      del self.model; del self.processor; self.model = None; self.processor = None
      gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
      return "Model unloaded."
    return "Not loaded."

  def _generate_stream(self, messages, max_px=1024):
    from qwen_vl_utils import process_vision_info
    from transformers import TextIteratorStreamer
    from queue import Empty
    if not self.processor or not self.model: yield {"text": "Error", "spinner": "❌"}; return
    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt", max_pixels=max_px*28*28).to(self.model.device)
    streamer = TextIteratorStreamer(self.processor, skip_special_tokens=True, skip_prompt=True)
    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=1024, do_sample=False, use_cache=True)
    thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs); thread.start()
    generated_text = ""; spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]; idx = 0
    while thread.is_alive() or not streamer.text_queue.empty():
      try:
        new_text = streamer.text_queue.get(timeout=0.05)
        if new_text == streamer.stop_signal: break
        generated_text += new_text
      except Empty: pass
      yield {"text": generated_text, "spinner": spinner_frames[idx % len(spinner_frames)]}; idx += 1
    yield {"text": generated_text, "spinner": "✅"}

  def _extract_json(self, text: str):
    try:
      m = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
      if m: return json.loads(m.group(1).replace("```json", "").replace("```", "").strip())
    except: pass
    return None

  def _get_category_schema(self, category: str, doc_type: str = ""):
    """Returns distinct, deeply commented schema blocks using structured formatting."""
    def F(desc, dtype="String", mode="FIND"): return {"desc": desc, "type": dtype, "mode": mode}
    
    # 1. Header
    base_header = {
      "document_type": F("CLASSIFIED TYPE", "String", "Info"), 
      "document_number": F("Primary Identifier", "String"), 
      "document_status": F("ORIGINAL, COPY, DRAFT", "String", "INFER"),
      "issue_date": F("Date of Creation (YYYY-MM-DD)", "String"),
      "reference_export": F("Export Ref, Shipper Ref", "String", "CAPTURE"),
      "page_current": F("Page X of Y", "Number", "FIND"),
      "page_total": F("Total Pages", "Number", "FIND")
    }
    
    if doc_type in ["CI", "PI", "PL"]:
      base_header["document_number"] = F("Invoice No, Commercial Invoice No", "String")
      base_header["reference_buyer"] = F("PO No, Purchase Order No, Contract No", "String")
      base_header["issue_date"] = F("Invoice Date", "String")
    elif doc_type in ["BL", "AWB", "AN"]:
      base_header["document_number"] = F("Bill of Lading No, AWB No, Reference No", "String")
      base_header["reference_carrier"] = F("Booking No, FMC No, Carrier Ref", "String")
      base_header["issue_date"] = F("Date of Issue (Not Sailing Date)", "String")
      base_header["bl_type"] = F("ORIGINAL, WAYBILL, SURRENDER", "String", "INFER")
      base_header["number_of_originals"] = F("No. of Original B/L", "Number", "FIND")
    elif doc_type == "LC":
      base_header["document_number"] = F("Documentary Credit No, L/C No", "String")
      base_header["expiry_date"] = F("Date of Expiry", "String")
    elif doc_type == "CO":
      base_header["document_number"] = F("Certificate No", "String")
        
    # 2. Parties
    parties = {
      "supplier_name": F("Sender Entity", "String"), 
      "supplier_address": F("Address of Supplier", "String"),
      "buyer_name": F("Receiver Entity", "String"), 
      "buyer_address": F("Address of Buyer", "String")
    }
    
    if doc_type in ["CI", "PI", "PL", "CO", "CERT"]:
      parties["supplier_name"] = F("Seller, Exporter", "String")
      parties["buyer_name"] = F("Buyer, Importer, Sold To", "String")
      parties["manufacturer_name"] = F("Producer/Manufacturer if different", "String", "Optional")
      parties["supplier_tax_id"] = F("VAT No, Tax ID", "String", "Optional")
      parties["buyer_tax_id"] = F("VAT No, Tax ID", "String", "Optional")
      parties["supplier_contact_info"] = F("Tel, Email, Fax", "String", "Optional")
      parties["buyer_contact_info"] = F("Tel, Email, Fax", "String", "Optional")
      
      if doc_type == "CI":
        parties["bank_name"] = F("Beneficiary Bank", "String", "Optional")
        parties["bank_account_number"] = F("Account No, IBAN", "String", "Optional")
        parties["bank_swift_code"] = F("SWIFT, BIC Code", "String", "Optional")
        parties["bank_address"] = F("Bank Address", "String", "Optional")

    elif doc_type in ["BL", "AWB", "AN"]:
      parties["supplier_name"] = F("Shipper", "String")
      parties["buyer_name"] = F("Consignee", "String")
      parties["notify_party_name"] = F("Notify Party Name", "String")
      parties["notify_party_address"] = F("Notify Party Address", "String", "Optional")
      parties["notify_party_contact"] = F("Notify Party Email/Phone", "String", "Optional")
      parties["issuer_name"] = F("Carrier Name / Agent Name", "String")
      parties["issuer_signature"] = F("Authorized Signature", "String", "Optional")
      parties["authentication_type"] = F("SIGNED, STAMPED", "String", "INFER")
      
      if doc_type == "AN":
        parties["delivery_agent_name"] = F("Delivery Agent", "String", "Optional")

    elif doc_type == "LC":
      parties["supplier_name"] = F("Beneficiary", "String")
      parties["buyer_name"] = F("Applicant", "String")
      parties["issuer_name"] = F("Issuing Bank", "String")
    elif doc_type == "INSURANCE":
      parties["surveyor_agent_name"] = F("Surveyor / Claims Agent", "String", "Optional")

    # 3. Logistics
    logistics = {
      "location_port_of_loading": F("POL, Airport of Departure", "String"),
      "location_port_of_discharge": F("POD, Airport of Destination", "String")
    }
    
    if doc_type in ["BL", "AWB", "AN", "PL"]:
      logistics["vehicle_name"] = F("Vessel Name, Flight No", "String")
      logistics["voyage_number"] = F("Voyage No", "String")
      logistics["departure_date"] = F("Shipped on Board Date", "String")
      logistics["arrival_date"] = F("ETA, Estimated Arrival", "String", "Optional")
      logistics["location_place_of_receipt"] = F("Place of Receipt", "String", "Optional")
      logistics["location_place_of_delivery"] = F("Place of Delivery", "String", "Optional")
      logistics["location_transshipment"] = F("Transshipment Port", "String", "Optional")
      logistics["move_type"] = F("FCL, LCL, CY/CY", "String", "Optional")
      logistics["transport_mode"] = F("SEA, AIR, ROAD", "String", "INFER")
      
      if doc_type == "AN":
        logistics["demurrage_free_days"] = F("Free Time", "String", "Optional")
      if doc_type == "BOOKING":
        logistics["cargo_closing_date"] = F("Cargo Cut-off", "String", "Optional")

    # 4. Conditions
    conditions = {}
    if doc_type in ["CI", "PI", "PL"]:
      conditions["incoterms_code"] = F("FOB, CIF, EXW, DDP", "String")
      conditions["incoterms_place"] = F("Location after Incoterm", "String", "Optional")
      conditions["payment_terms_type"] = F("T/T, L/C", "String")
      conditions["payment_terms_days"] = F("Net 30, 60 Days", "String", "Optional")
    elif doc_type in ["BL", "AWB"]:
      conditions["freight_payment_term"] = F("Freight Prepaid, Freight Collect", "String")
      conditions["freight_payable_at"] = F("Payable at location", "String", "Optional")
    elif doc_type == "LC":
      conditions["lc_tenor"] = F("At Sight, Usance", "String")
      conditions["lc_tolerance"] = F("Tolerance (e.g. 10/10)", "String", "Optional")
      conditions["presentation_period"] = F("Presentation Period", "String", "Optional")
    elif doc_type == "CO":
      conditions["origin_criterion"] = F("Wholly Obtained, PSR", "String")
      conditions["cumulation_applied"] = F("Cumulation Checkbox", "String", "Optional")
    elif doc_type == "INSURANCE":
      conditions["insurance_condition"] = F("All Risks, ICC(A)", "String")
    elif doc_type == "CERT":
      conditions["inspection_standard"] = F("SGS, ISO 9001", "String", "Optional")

    # 5. Financials
    financials = {
      "currency_code": F("Currency Symbol or Code (USD, EUR)", "String")
    }
    if doc_type in ["CI", "PI", "AN", "LC", "INSURANCE", "AWB"]:
      amt_desc = "Grand Total Amount (MUST have currency symbol)"
      if doc_type in ["CI", "PI"]: amt_desc = "CRITICAL: Final Invoice Payable Amount"
      
      financials["amount_total"] = F(amt_desc, "Number")
      
      if doc_type == "CI":
        financials["amount_subtotal"] = F("Subtotal", "Number")
        financials["amount_tax"] = F("VAT, Tax amount", "Number")
        financials["charge_freight"] = F("Freight Amount", "Number")
        financials["charge_insurance"] = F("Insurance Cost", "Number", "Optional")
      elif doc_type == "AN":
        financials["charge_local"] = F("Sum of Local Charges (THC, DOC)", "Number")
        financials["exchange_rate"] = F("Exchange Rate", "Number", "Optional")
        financials["exchange_rate_date"] = F("Exchange Rate Date", "String", "Optional")
      elif doc_type == "INSURANCE":
        financials["value_insured"] = F("Insured Amount", "Number")
        financials["charge_insurance"] = F("Premium Amount", "Number")
      elif doc_type == "AWB":
        financials["value_declared_customs"] = F("Declared Value for Customs", "String", "Optional")
        financials["value_declared_carriage"] = F("Declared Value for Carriage", "String", "Optional")

    # 6. Cargo
    cargo = {
      "package_count": F("Total Quantity (NOT Money)", "Number"),
      "weight_gross": F("Total Gross Weight (NOT Money)", "Number")
    }
    if doc_type in ["BL", "AWB", "PL", "MSDS"]:
      cargo["package_unit"] = F("Unit (CTN, PKGS, PLT)", "String")
      cargo["weight_net"] = F("Total Net Weight", "Number")
      cargo["weight_gross_unit"] = F("Unit (KG, LBS)", "String")
      cargo["volume_measurement"] = F("Total Volume (CBM)", "Number")
      cargo["volume_unit"] = F("Unit (CBM)", "String")
      cargo["marks_and_numbers"] = F("Marks & Nos", "String", "CAPTURE")
      
      if doc_type in ["BL", "MSDS"]:
        cargo["dg_un_number"] = F("UN Number (Dangerous Goods)", "String", "Optional")
        cargo["dg_class_code"] = F("IMO Class", "String", "Optional")
        cargo["dg_packing_group"] = F("Packing Group", "String", "Optional")
        cargo["dg_flash_point"] = F("Flash Point", "String", "Optional")

    # 7. Items & Containers
    line_items_obj = {
      "sequence_number": F("1, 2, 3...", "Number"),
      "description": F("Description of Goods", "String"), 
      "quantity": F("Line Item Quantity", "Number"),
      "quantity_unit": F("Unit (PCS, SET)", "String")
    }
    if doc_type in ["CI", "PI", "CO", "PL"]:
      line_items_obj["hs_code"] = F("HS Code / Tariff No", "String")
      line_items_obj["origin_country"] = F("Country of Origin", "String", "Optional")
      
      if doc_type in ["CI", "PI"]:
        line_items_obj["price_unit"] = F("Unit Price", "Number")
        line_items_obj["price_total"] = F("Line Total Amount", "Number")
        line_items_obj["is_sample"] = F("True if Sample/No Value", "Boolean", "INFER")
      elif doc_type == "PL":
        line_items_obj["weight_net"] = F("Net Weight", "Number", "Optional")
        line_items_obj["weight_gross"] = F("Gross Weight", "Number", "Optional")
        line_items_obj["volume_measurement"] = F("Volume", "Number", "Optional")
        line_items_obj["package_quantity"] = F("Package Qty", "Number", "Optional")
    
    container_obj = {
      "container_number": F("Container No (4 char + 7 digit)", "String")
    }
    if doc_type in ["BL", "PL"]:
      container_obj["seal_number"] = F("Seal No", "String")
      container_obj["type_description"] = F("Type (20GP, 40HC)", "String")
      container_obj["iso_type_code"] = F("ISO Code", "String", "Optional")
      container_obj["package_count"] = F("Pkg Count in Container", "Number", "Optional")
      container_obj["weight_total"] = F("Container Weight", "Number", "Optional")

    # Formatter for logic.py (needs to return JSON string block)
    def fmt(d, name, is_list=False):
      lines = [f'  "{name}": [{{' if is_list else f'  "{name}": {{']
      items = d.items()
      for i, (k, v) in enumerate(items):
        dtype = v.get("type", "String")
        desc = v.get("desc", "")
        mode = v.get("mode", "FIND")
        default_val = "0" if dtype == "Number" else "false" if dtype == "Boolean" else '""'
        comma = "," if i < len(items) - 1 else ""
        lines.append(f'    "{k}": {default_val}{comma} // {mode}: {desc} {{{dtype}}}.')
      lines.append('  }]' if is_list else '  }')
      return "\n".join(lines)

    # Return only the requested category block
    if category == "header": return f'{{\n{fmt(base_header, "header")}\n}}'
    if category == "parties": return f'{{\n{fmt(parties, "parties")}\n}}'
    if category == "logistics": return f'{{\n{fmt(logistics, "logistics")}\n}}'
    if category == "conditions": return f'{{\n{fmt(conditions, "conditions")}\n}}'
    if category == "financials": return f'{{\n{fmt(financials, "financials")}\n}}'
    if category == "cargo": return f'{{\n{fmt(cargo, "cargo")}\n}}'
    if category == "items": return f'{{\n{fmt(line_items_obj, "line_items", is_list=True)}\n}}'
    if category == "containers": return f'{{\n{fmt(container_obj, "containers", is_list=True)}\n}}'
    return "{}"

  def _json_to_md(self, category, data):
    if not data: return ""
    md = ""
    try:
      if category == "parties" and isinstance(data, dict):
        if any(isinstance(v, dict) for v in data.values()):
          md += "| Party | Role | Name | Address |\n| :--- | :--- | :--- | :--- |\n"
          for key, val in data.items():
            if isinstance(val, dict):
              role = str(val.get("role") or "")
              name = str(val.get("name") or "").replace("\n", " ")
              addr = str(val.get("address") or "").replace("\n", " ")[:50]
              md += f"| {key.title()} | {role} | {name} | {addr} |\n"
          return md
      target_list = None
      if isinstance(data, list): target_list = data
      elif isinstance(data, dict):
        for k, v in data.items():
          if isinstance(v, list): target_list = v; break
      if target_list:
        if not target_list: return "(No items found)"
        keys = list(target_list[0].keys())
        headers = [k.replace("_", " ").title() for k in keys]
        md += "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join([":---"] * len(keys)) + " |\n"
        for item in target_list:
          row = [str(item.get(k) or "").replace("\n", " ") for k in keys]
          md += "| " + " | ".join(row) + " |\n"
        return md
      if isinstance(data, dict):
        if len(data) == 1 and isinstance(list(data.values())[0], dict):
          data = list(data.values())[0]
        md += "| Field | Value |\n| :--- | :--- |\n"
        for k, v in data.items():
          if isinstance(v, (dict, list)): continue
          val_str = str(v).replace("\n", " ")
          md += f"| {k.replace('_', ' ').title()} | {val_str} |\n"
        return md
    except Exception: return "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```"
    return "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```"

  def _generate_rich_summary(self, doc_type: str, data: Dict[str, Any]) -> str:
    type_map = {"CI": "Commercial Invoice", "PI": "Proforma Invoice", "PL": "Packing List", "BL": "Bill of Lading", "AWB": "Air Waybill", "CO": "Certificate of Origin", "LC": "Letter of Credit"}
    full_type = type_map.get(doc_type, doc_type)
    parts = [f"This is a {full_type} document."]
    h = data.get("header", {}); doc_no = h.get("document_number")
    if doc_no and doc_no not in ["N/A", "Unknown", ""]: parts.append(f"Document number is {doc_no}.")
    date = h.get("issue_date")
    if date and date not in ["N/A", "Unknown", ""]: parts.append(f"Issued on {date}.")
    p = data.get("parties", {}); supplier = p.get("supplier_name"); buyer = p.get("buyer_name")
    clean_sup = supplier if supplier and supplier not in ["N/A", "Unknown"] else None
    clean_buy = buyer if buyer and buyer not in ["N/A", "Unknown"] else None
    if clean_sup and clean_buy: parts.append(f"Transaction involved {clean_sup} as the supplier/shipper and {clean_buy} as the buyer/consignee.")
    elif clean_sup: parts.append(f"Supplier/Shipper is {clean_sup}.")
    elif clean_buy: parts.append(f"Buyer/Consignee is {clean_buy}.")
    f = data.get("financials", {}); amt = f.get("amount_total"); curr = f.get("currency_code", "USD")
    if amt and str(amt) != "0.0": parts.append(f"Total amount is {amt} {curr}.")
    l = data.get("logistics", {}); pol = l.get("location_port_of_loading"); pod = l.get("location_port_of_discharge"); mode = l.get("transport_mode")
    if pol and pod: parts.append(f"Shipped from {pol} to {pod}.")
    if mode: parts.append(f"Transport mode is {mode}.")
    items = data.get("line_items", [])
    if items:
      item_descs = []
      for it in items[:5]:
        d = it.get("description")
        if d and len(d) > 3: item_descs.append(d)
      if item_descs: parts.append(f"Contains items: {', '.join(item_descs)}.")
    return " ".join(parts)

  def process_image(self, image_path: str, low_vram: bool = False, do_save: bool = True):
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]; s_idx = [0]
    def get_spinner(): s = spinner_frames[s_idx[0] % len(spinner_frames)]; s_idx[0] += 1; return s
    all_raw_thoughts = "AI Analyzing...\n\n"
    if self.model is None:
      t = threading.Thread(target=self._load_main_model); t.start()
      while t.is_alive():
        yield {"json": None, "summary": "", "raw": all_raw_thoughts + f"{get_spinner()} Starting AI Engine...", "spinner": ""}
        time.sleep(0.1)
    if self.model is None: yield {"json": None, "summary": "Model failed to load.", "raw": all_raw_thoughts + "❌ Model failed to load.", "spinner": "❌"}; return
    all_raw_thoughts += "🚀 AI Engine Ready.\n\n"
    try:
      full_img = Image.open(image_path)
      yield {"json": None, "summary": "", "raw": all_raw_thoughts + f"{get_spinner()} Identifying document type...", "spinner": ""}
      class_img = full_img.copy(); class_img.thumbnail((768, 768))
      msg_type = [{"role": "user", "content": [{"type": "image", "image": class_img}, {"type": "text", "text": "Classify document: PI, CI, BL, AWB, PL. Return JSON: {'doc_type': ''}"}]}]
      type_res = ""
      for p in self._generate_stream(msg_type, max_px=768):
        type_res = p["text"]
        yield {"json": None, "summary": "", "raw": all_raw_thoughts + f"{p['spinner']} Identifying document type...", "spinner": ""}
      clean_type_res = type_res.replace("```json", "").replace("```", "").strip()
      detected_type = json.loads(clean_type_res[clean_type_res.find("{"):clean_type_res.rfind("}")+1]).get("doc_type", "Unknown")
    except: detected_type = "Unknown"
    all_raw_thoughts += f"✅ Document identified: **{detected_type}**\n\n"
    SLICE_CONFIG = {
      "CI": {"missions": [{"cat": "header", "box": (0.00, 0.20)}, {"cat": "parties", "box": (0.00, 0.40)}, {"cat": "logistics", "box": (0.20, 0.50)}, {"cat": "items", "box": (0.30, 0.70)}, {"cat": "items", "box": (0.50, 0.85)}, {"cat": "financials", "box": (0.70, 0.95)}, {"cat": "conditions", "box": (0.80, 1.00)}]},
      "PI": {"missions": [{"cat": "header", "box": (0.00, 0.20)}, {"cat": "parties", "box": (0.00, 0.40)}, {"cat": "logistics", "box": (0.20, 0.50)}, {"cat": "items", "box": (0.30, 0.70)}, {"cat": "items", "box": (0.50, 0.85)}, {"cat": "financials", "box": (0.70, 0.95)}, {"cat": "conditions", "box": (0.80, 1.00)}]},
      "PL": {"missions": [{"cat": "header", "box": (0.00, 0.20)}, {"cat": "parties", "box": (0.00, 0.40)}, {"cat": "logistics", "box": (0.20, 0.50)}, {"cat": "items", "box": (0.30, 0.80)}, {"cat": "cargo", "box": (0.60, 0.95)}, {"cat": "conditions", "box": (0.85, 1.00)}]},
      "BL": {"missions": [{"cat": "header", "box": (0.00, 0.20)}, {"cat": "parties", "box": (0.00, 0.60)}, {"cat": "logistics", "box": (0.35, 0.65)}, {"cat": "cargo", "box": (0.50, 0.90)}, {"cat": "conditions", "box": (0.80, 1.00)}]},
      "AWB": {"missions": [{"cat": "header", "box": (0.00, 0.15)}, {"cat": "parties", "box": (0.00, 0.40)}, {"cat": "logistics", "box": (0.10, 0.40)}, {"cat": "cargo", "box": (0.30, 0.70)}, {"cat": "financials", "box": (0.60, 0.90)}, {"cat": "conditions", "box": (0.85, 1.00)}]},
      "CO": {"missions": [{"cat": "header", "box": (0.00, 0.20)}, {"cat": "parties", "box": (0.00, 0.40)}, {"cat": "logistics", "box": (0.30, 0.50)}, {"cat": "items", "box": (0.40, 0.80)}, {"cat": "conditions", "box": (0.75, 1.00)}]},
      "LC": {"missions": [{"cat": "header", "box": (0.00, 0.30)}, {"cat": "parties", "box": (0.00, 0.40)}, {"cat": "financials", "box": (0.20, 0.50)}, {"cat": "logistics", "box": (0.40, 0.70)}, {"cat": "conditions", "box": (0.60, 1.00)}]}
    }
    cfg = SLICE_CONFIG.get(detected_type, {"missions": [{"cat": "header", "box": (0.0, 0.3)}, {"cat": "items", "box": (0.2, 0.8)}, {"cat": "conditions", "box": (0.7, 1.0)}]})
    missions = cfg["missions"]; num_slices = len(missions)
    all_raw_thoughts += f"🎯 Starting {num_slices}-step strategic scan for {detected_type}...\n\n"
    w, h = full_img.size
    final_data = {"header": {"doc_type": detected_type}, "parties": {}, "logistics": {}, "conditions": {}, "financials": {}, "cargo": {}, "line_items": [], "containers": []}
    for i, mission in enumerate(missions):
      category = mission["cat"]; top_pct, bot_pct = mission["box"]
      start_y, end_y = int(h * top_pct), int(h * bot_pct)
      img_slice = full_img.crop((0, start_y, w, end_y))
      schema = self._get_category_schema(category, detected_type)
      schema = "\n".join([line.strip() for line in schema.split("\n") if line.strip()])
      task_desc = f"[{detected_type}] {category.upper()} ({int(top_pct*100)}%~{int(bot_pct*100)}%)"
      yield {"json": None, "summary": f"Analyzing: {task_desc}", "raw": all_raw_thoughts + f"{get_spinner()} Analyzing: {task_desc}...", "spinner": get_spinner()}
      prompt = f"RULES: Follow comments strictly. Output JSON ONLY. MISSION: Extract data for category '{category.upper()}'.\nSCHEMA:\n{schema}"
      messages = [{"role": "user", "content": [{"type": "image", "image": img_slice}, {"type": "text", "text": prompt}]}]
      tile_res = ""
      for p in self._generate_stream(messages, max_px=1024):
        tile_res = p["text"]
        yield {"json": None, "summary": f"Processing: {task_desc}", "raw": all_raw_thoughts + f"{p['spinner']} Analyzing: {task_desc}...\n{tile_res}", "spinner": p["spinner"]}
      md_table = tile_res
      try:
        clean_res = tile_res.replace("```json", "").replace("```", "").strip()
        sj = clean_res.find("{"); ej = clean_res.rfind("}")
        if sj != -1 and ej != -1:
          data = json.loads(clean_res[sj:ej+1])
          display_data = data
          if category == "items" and "line_items" in data: display_data = data["line_items"]
          elif category == "containers" and "containers" in data: display_data = data["containers"]
          elif category in data and isinstance(data[category], (dict, list)): display_data = data[category]
          md_table = self._json_to_md(category, display_data)
          if category == "items" and "line_items" in data:
            for item in data["line_items"]:
              d = (item.get("description") or "").strip()
              if len(d) > 3 and not any(d[:15] in (ex.get("description") or "") for ex in final_data["line_items"]):
                final_data["line_items"].append(item)
          elif category == "containers" and "containers" in data:
            for c in data["containers"]:
              cn = c.get("container_number")
              if cn and not any(cn == ex.get("container_number") for ex in final_data["containers"]):
                final_data["containers"].append(c)
          else:
            extracted_data = data.get(category) or data
            if isinstance(extracted_data, dict):
              if category not in final_data: final_data[category] = {}
              if isinstance(final_data[category], dict):
                final_data[category].update({sk:sv for sk,sv in extracted_data.items() if sv and sv not in ["N/A", "Unknown", ""]})
      except: pass
      all_raw_thoughts += f"Analyzing: {task_desc}..\n{md_table}\n\n\n"
    h = final_data["header"]; comp_name = final_data["parties"].get("supplier_name") or final_data["parties"].get("buyer_name") or "Unknown"
    summary_text = self._generate_rich_summary(detected_type, final_data)
    issue_date = h.get("issue_date", "")
    try: total_amount = float(final_data.get("financials", {}).get("amount_total", 0.0))
    except: total_amount = 0.0
    l = final_data.get("logistics", {}); vessel = l.get("vehicle_name") or l.get("vehicle_name", ""); pol = l.get("location_port_of_loading", ""); pod = l.get("location_port_of_discharge", ""); incoterms = final_data.get("conditions", {}).get("incoterms_code", "")
    record = {"uuid": str(uuid.uuid4()), "vector": self._get_embedding(summary_text), "text": summary_text, "id": str(h.get('document_number','N/A')), "no": "N/A", "name": str(comp_name), "type": str(detected_type), "issue_date": str(issue_date), "total_amount": total_amount, "vessel": str(vessel), "pol": str(pol), "pod": str(pod), "incoterms": str(incoterms), "data": json.dumps(final_data)}
    if do_save:
      try:
        tbl = self.db.open_table(self.table_name); tbl.add([record])
      except:
        try: self.db.create_table(self.table_name, data=[record])
        except:
          self.db.drop_table(self.table_name); self.db.create_table(self.table_name, data=[record])
    yield {"json": final_data, "summary": summary_text, "raw": all_raw_thoughts, "spinner": "✅"}
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

  def list_documents(self):
    tbl = self._get_table();
    if not tbl: return []
    df = tbl.search().limit(100).to_pandas()
    return [[r['uuid'], r['type'], r['name'], r['id'], r.get('issue_date', ''), r.get('total_amount', 0.0), r.get('vessel', ''), r.get('pol', ''), r.get('pod', ''), r.get('incoterms', ''), r['text']] for _, r in df.iterrows()]

  def search_documents(self, query: str):
    tbl = self._get_table(); emb = self._get_embedding(query)
    try: return [[r['uuid'], r['type'], r['name'], r['id'], r.get('issue_date', ''), r.get('total_amount', 0.0), r['text']] for _, r in tbl.search(emb).metric("cosine").limit(10).to_pandas().iterrows()]
    except: return []

  def filter_documents(self, query: str = "", doc_type: str = "All", min_amt: float = 0.0, max_amt: float = 0.0, date_from: str = "", date_to: str = "", vessel: str = "", pol: str = "", pod: str = "", incoterms: str = ""):
    tbl = self._get_table()
    if not tbl: return []
    
    conditions = []
    if doc_type and doc_type != "All":
      conditions.append(f"type = '{doc_type}'")
    if min_amt > 0:
      conditions.append(f"total_amount >= {min_amt}")
    if max_amt > 0:
      conditions.append(f"total_amount <= {max_amt}")
    if date_from:
      conditions.append(f"issue_date >= '{date_from}'")
    if date_to:
      conditions.append(f"issue_date <= '{date_to}'")
    
    if vessel:
      conditions.append(f"vessel LIKE '%{vessel}%'")
    if pol:
      conditions.append(f"pol LIKE '%{pol}%'")
    if pod:
      conditions.append(f"pod LIKE '%{pod}%'")
    if incoterms:
      conditions.append(f"incoterms = '{incoterms}'")
      
    where_clause = " AND ".join(conditions) if conditions else None
    
    try:
      if query and query.strip():
        emb = self._get_embedding(query)
        search = tbl.search(emb).metric("cosine")
        if where_clause: search = search.where(where_clause)
        df = search.limit(50).to_pandas()
      else:
        search = tbl.search()
        if where_clause: search = search.where(where_clause)
        df = search.limit(100).to_pandas()
        
      return [[r['uuid'], r['type'], r['name'], r['id'], r.get('issue_date', ''), r.get('total_amount', 0.0), r.get('vessel', ''), r.get('pol', ''), r.get('pod', ''), r.get('incoterms', ''), r['text']] for _, r in df.iterrows()]
    except Exception as e: print(f"Filter Error: {e}"); return []

  def advanced_hybrid_search(self, query: str, filters: Dict[str, Any], doc_type: str = "All"):
    """True Hybrid Search: Maps AI hierarchical paths to flat DB columns and combines with Vector search."""
    tbl = self._get_table()
    if not tbl: return []
    
    PATH_MAP = {
      "header.document_number": "id",
      "header.issue_date": "issue_date",
      "parties.supplier_name": "name",
      "parties.buyer_name": "name",
      "financials.amount_total": "total_amount",
      "logistics.vehicle_name": "vessel",
      "logistics.location_port_of_loading": "pol",
      "logistics.location_port_of_discharge": "pod",
      "conditions.incoterms_code": "incoterms"
    }
    
    conditions = []
    if doc_type and doc_type not in ["All", "GENERIC", "UNKNOWN"]:
      conditions.append(f"type = '{doc_type}'")
      
    for path, cond in filters.items():
      db_col = PATH_MAP.get(path, path)
      
      try:
        valid_cols = ["id", "name", "total_amount", "vessel", "pol", "pod", "incoterms", "issue_date", "type"]
        if db_col.split('.')[0] not in valid_cols and db_col not in valid_cols: continue

        if isinstance(cond, dict):
          op = list(cond.keys())[0]
          val = cond[op]
          if op == "$gt": conditions.append(f"{db_col} > {val}")
          elif op == "$lt": conditions.append(f"{db_col} < {val}")
          elif op == "$eq": conditions.append(f"{db_col} = '{val}'" if isinstance(val, str) else f"{db_col} = {val}")
          elif op == "$contains": conditions.append(f"{db_col} LIKE '%{val}%'")
        else:
          conditions.append(f"{db_col} LIKE '%{cond}%'")
      except: continue
      
    where_clause = " AND ".join(conditions) if conditions else None
    
    try:
      search = tbl.search(self._get_embedding(query)).metric("cosine")
      if where_clause: search = search.where(where_clause)
      
      df = search.limit(50).to_pandas()
      return [[r['uuid'], r['type'], r['name'], r['id'], r.get('issue_date', ''), r.get('total_amount', 0.0), r.get('vessel', ''), r.get('pol', ''), r.get('pod', ''), r.get('incoterms', ''), r['text']] for _, r in df.iterrows()]
    except Exception as e: print(f"Hybrid Search Error: {e}"); return []

  def get_document(self, doc_id):
    tbl = self._get_table(); res = tbl.search().where(f"uuid = '{doc_id}'").limit(1).to_list() if tbl else []
    if not res: return None, None
    return {"id": res[0]['id'], "type": res[0]['type'], "no": res[0]['no'], "data": res[0]['data']}, res[0]['text']

  def delete_document(self, doc_id):
    tbl = self._get_table(); tbl.delete(f"uuid = '{doc_id}'") if tbl else None
    return "Deleted"

  def update_document(self, doc_id, new_json_str):
    tbl = self._get_table(); tbl.update(where=f"uuid = '{doc_id}'", values={"data": new_json_str}) if tbl else None
    return "Updated"