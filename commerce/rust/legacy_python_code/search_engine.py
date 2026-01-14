import json
import threading
import re
from queue import Empty

class TradeSearchEngine:
    def __init__(self, model, processor, extract_json_fn, get_schema_fn):
        """
        Args:
            model: The loaded HF model (Qwen2-VL)
            processor: The loaded processor
            extract_json_fn: Helper function to parse JSON from text
            get_schema_fn: Helper function (unused here, we define search-specific schema locally)
        """
        self.model = model
        self.processor = processor
        self.extract_json = extract_json_fn
        # Note: get_schema_fn from app.py is _get_category_schema (for extraction), 
        # but for search we need a flattened schema map, so we define it internally.

    def _generate_stream(self, messages, max_new_tokens=1024):
        """Internal streaming generation helper."""
        from transformers import TextIteratorStreamer
        from qwen_vl_utils import process_vision_info

        if not self.processor or not self.model:
            yield {"text": "Error: Model not loaded", "spinner": "❌"}
            return

        # Prepare inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            max_pixels=1024*28*28 # Use high res for text clarity if needed, though mostly text-based here
        ).to(self.model.device)

        streamer = TextIteratorStreamer(self.processor, skip_special_tokens=True, skip_prompt=True)
        generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0

        while thread.is_alive() or not streamer.text_queue.empty():
            try:
                new_text = streamer.text_queue.get(timeout=0.05)
                if new_text == streamer.stop_signal:
                    break
                generated_text += new_text
            except Empty:
                pass
            
            yield {
                "text": generated_text, 
                "spinner": spinner_frames[idx % len(spinner_frames)]
            }
            idx += 1
            
        yield {"text": generated_text, "spinner": "✅"}

    def _get_search_schema_definitions(self, doc_type="ALL"):
        """Defines the mapping between natural language concepts and DB columns."""
        def F(desc, dtype="String", mode="FIND"): return f"{{ 'desc': '{desc}', 'type': '{dtype}' }}"
        
        # Simplified schema for Search Prompt
        schema = {
            "header": {
                "document_type": F("Type (Invoice, BL, AWB...)", "String"),
                "document_number": F("ID, Doc No, Reference No", "String"),
                "issue_date": F("Date (YYYY-MM-DD)", "String")
            },
            "parties": {
                "supplier_name": F("Seller, Shipper, Exporter", "String"),
                "buyer_name": F("Buyer, Consignee, Importer", "String")
            },
            "financials": {
                "amount_total": F("Total Value/Amount", "Number")
            },
            "logistics": {
                "vehicle_name": F("Vessel Name, Flight No", "String"),
                "location_port_of_loading": F("POL, Origin", "String"),
                "location_port_of_discharge": F("POD, Destination", "String")
            },
            "conditions": {
                "incoterms_code": F("Incoterms (FOB, CIF)", "String")
            }
        }
        
        # Flatten for prompt
        flat_defs = []
        for cat, fields in schema.items():
            for field, meta in fields.items():
                flat_defs.append(f'"{cat}.{field}": {meta}')
                
        return "{\n" + "\n".join(flat_defs) + "\n}"

    def parse_query(self, query):
        """
        Main entry point for AI Search. Yields progress steps.
        Returns final data in the last yield with step='complete'.
        """
        
        # 1. Split Contexts
        yield {"step": "thinking", "message": "Analyzing query structure...", "spinner": "🧠"}
        
        system_prompt = "Split user query into JSON sub-queries with document type. Structure: [{"query": "...", "header": {"document_type": "TYPE"}}]]"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\nJSON Output:"}
        ]
        
        full_text = ""
        for chunk in self._generate_stream(messages, max_new_tokens=512):
            full_text = chunk["text"]
            # Optional: yield streaming text if we wanted verbose mode
        
        sub_contexts = self.extract_json(full_text)
        if not isinstance(sub_contexts, list):
            sub_contexts = [{"query": query, "header": {"document_type": "ALL"}}]

        final_plans = []
        
        # 2. Process each sub-context
        for i, ctx in enumerate(sub_contexts):
            sub_q = ctx.get("query", query)
            doc_type = ctx.get("header", {}).get("document_type", "ALL")
            
            yield {"step": "processing", "message": f"Building filters for '{sub_q}' ({doc_type})...", "spinner": "🔍"}
            
            # Prepare Search Prompt
            schema_def = self._get_search_schema_definitions(doc_type)
            prompt = f"""Extract filters into JSON. 
RULES: 
1. NO HALLUCINATION. 
2. PRECISE MAPPING to Schema. 
3. OPERATORS: \"$eq\", \"$contains\", \"$gt\" (greater), \"$lt\" (less), \"$gte\", \"$lte\".

DYNAMIC SCHEMA:
{schema_def}

REQUIRED JSON FORMAT: 
{{
  "header": {{ "document_type": "{doc_type}" }},
  "filters": {{
      "category.field_path": {{ "$operator": value }}
  }}
}} """
            
            messages = [
                {"role": "system", "content": prompt}, 
                {"role": "user", "content": f"Query: {sub_q}\nJSON Output:"}
            ]
            
            filter_text = ""
            for chunk in self._generate_stream(messages, max_new_tokens=512):
                filter_text = chunk["text"]

            parsed = self.extract_json(filter_text)
            
            if parsed:
                # Clean up filters
                raw_filters = parsed.get("filters", {})
                clean_filters = {}
                for k, v in raw_filters.items():
                    # Remove $ prefixes from keys if AI added them mistakenly
                    clean_key = k.replace("$filter.", "").replace("$$", "") # basic cleanup
                    clean_filters[clean_key] = v
                
                final_plans.append({
                    "query": sub_q,
                    "type": parsed.get("header", {}).get("document_type", "ALL"),
                    "filter": clean_filters
                })
            else:
                # Fallback for parsing failure
                final_plans.append({
                    "query": sub_q,
                    "type": doc_type,
                    "filter": {}
                })

        yield {"step": "complete", "message": "Search planning complete.", "data": final_plans, "spinner": "✅"}