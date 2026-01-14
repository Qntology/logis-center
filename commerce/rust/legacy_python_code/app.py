import gradio as gr
import json
import pandas as pd
from logic import TradeLogisSystem
from search_engine import TradeSearchEngine

# Initialize System
system = TradeLogisSystem()
search_engine = None

def get_search_engine():
    global search_engine
    if search_engine is None:
        if system.model is None: system._load_main_model()
        # Pass the original schema logic to the new engine
        search_engine = TradeSearchEngine(system.model, system.processor, system._extract_json, system._get_category_schema)
    return search_engine

def handle_upload(image, low_vram_mode, do_save=True):
    if image is None: 
        yield "## ⚠️ No image uploaded.", "⚠️ No image"
        return

    # 1. Immediate Feedback
    yield "### ⏳ Initializing AI Model & Processing Image... Please wait.", "⏳ Initializing..."

    temp_path = image if isinstance(image, str) else "temp_upload.jpg"
    if not isinstance(image, str): image.save(temp_path)
    
    for result in system.process_image(temp_path, low_vram=low_vram_mode, do_save=do_save):
        spinner = result.get("spinner", "⏳")
        status_text = f"{spinner} AI Analyzing..."
        if result["json"] is None:
            yield result['raw'], status_text
        else:
            save_msg = " (Saved to DB)" if do_save else " (View Only)"
            json_str = json.dumps(result['json'], indent=2, ensure_ascii=False)
            # Construct formatted output safely
            formatted_output = f"""# ✅ Analysis Complete{save_msg}

### 🧠 Thinking Process
<details open>
<summary>Click to hide/view raw thought process</summary>

{result['raw']}
</details>

---

### 📄 Extracted JSON
```json
{json_str}
```

---

### 📝 Summary
{result['summary']}
"""
            yield formatted_output, "✅ Done"

def handle_extract(image, low_vram_mode):
    yield from handle_upload(image, low_vram_mode, do_save=True)

def handle_process(image, low_vram_mode):
    yield from handle_upload(image, low_vram_mode, do_save=True)

def change_device_handler(mode):
    return system.set_device_mode(mode)

def trigger_unload():
    return system.unload_model()

def refresh_list():
    docs = system.list_documents()
    return pd.DataFrame(docs, columns=["UUID", "Type", "Name", "ID", "Date", "Amount", "Vessel", "POL", "POD", "Incoterms", "Summary"])

def handle_search(query, doc_type, min_amt, max_amt, d_from, d_to, spec_1, spec_2, spec_3):
    # Map special inputs back to DB fields based on doc_type
    vessel, pol, pod, incoterms = "", "", "", ""
    if doc_type in ["Bill of Lading", "Air Waybill"]:
        vessel, pol, pod = spec_1, spec_2, spec_3
    elif doc_type in ["Commercial Invoice", "Proforma Invoice"]:
        incoterms = spec_1
        
    docs = system.filter_documents(query, doc_type, min_amt, max_amt, d_from, d_to, vessel, pol, pod, incoterms)
    return pd.DataFrame(docs, columns=["UUID", "Type", "Name", "ID", "Date", "Amount", "Vessel", "POL", "POD", "Incoterms", "Summary"])

def handle_type_change(doc_type):
    # Returns updates for spec_1, spec_2, spec_3 (Label, Value, Visibility)
    if doc_type in ["Bill of Lading", "Air Waybill"]:
        v_label = "Vessel Name" if doc_type == "Bill of Lading" else "Flight No"
        p_label = "Port of Loading" if doc_type == "Bill of Lading" else "Departure Airport"
        d_label = "Port of Discharge" if doc_type == "Bill of Lading" else "Arrival Airport"
        return gr.update(label=v_label, visible=True), gr.update(label=p_label, visible=True), gr.update(label=d_label, visible=True)
    elif doc_type in ["Commercial Invoice", "Proforma Invoice"]:
        return gr.update(label="Incoterms (FOB, CIF...)", visible=True), gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def load_details(evt: gr.SelectData, df, current_id):
    if df is None or len(df) == 0: return None, gr.update(), gr.update(), None, ""
    # UUID is at index 0
    doc_id = df.iloc[evt.index[0]].iloc[0]
    if str(doc_id) == str(current_id): return gr.update(), gr.update(), gr.update(), current_id, ""
    meta, summary = system.get_document(doc_id)
    if meta:
        try:
            # logic.py renamed full_json to data
            js = json.dumps(json.loads(meta.get("data", "{}")), indent=2, ensure_ascii=False)
            return doc_id, js, summary, doc_id, ""
        except:
            return doc_id, meta.get("data", "{}"), summary, doc_id, ""
    return None, "", "", None, ""

def save_changes(doc_id, json_input):
    return system.update_document(doc_id, json_input) if doc_id else "No selection."

def delete_record(doc_id):
    return system.delete_document(doc_id) if doc_id else "No selection."

def handle_nosql_search(query):
    if not query:
        yield pd.DataFrame(), "{}", "## ⚠️ Please enter a query."
        return
        
    engine = get_search_engine()
    status_history = "### 🤖 AI Search Analysis\n\n"
    
    final_plans = []
    for step in engine.parse_query(query):
        msg = step.get("message", "")
        spinner = step.get("spinner", "⏳")
        
        if step["step"] == "complete":
            final_plans = step.get("data", [])
            status_history += f"**{msg}**\n"
        else:
            yield gr.update(), gr.update(), status_history + f"{spinner} {msg}"
            continue

    aggregated_docs = []
    for plan in final_plans:
        # Use the NEW advanced_hybrid_search to handle all AI-extracted filters
        docs = system.advanced_hybrid_search(
            query=query, # Original natural language query for vector search
            filters=plan.get('filter', {}), # All AI-extracted NoSQL filters
            doc_type=plan.get('type') # Target document type
        )
        aggregated_docs.extend(docs)
        
    df_res = pd.DataFrame(aggregated_docs, columns=["UUID", "Type", "Name", "ID", "Date", "Amount", "Vessel", "POL", "POD", "Incoterms", "Summary"])
    yield df_res, json.dumps(final_plans, indent=2), status_history

with gr.Blocks(title="Trade Logistics AI Center") as demo:
    gr.Markdown("# 🚢 Trade Logistics AI Center")
    current_id_state = gr.State(None)

    with gr.Tabs():
        # --- TAB 1: UPLOAD ---
        with gr.Tab("STEP 1: Upload"):
            # ... (content remains same)
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="filepath", label="Upload")
                    # Toggle Button Logic
                    extract_btn = gr.Button("Extract & Save", variant="primary", scale=1)
                    cancel_btn = gr.Button("Cancel Extraction", variant="stop", scale=1, visible=False)
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Result"):
                            result_output = gr.Markdown(label="Analysis Result", height=600)

        # --- TAB 2: LIST ---
        with gr.Tab("STEP 2: List") as doc_tab:
             # ... (content remains same)
            with gr.Accordion("🔍 Advanced Search & Filters", open=True):
                with gr.Row():
                    search_query = gr.Textbox(label="Keywords", placeholder="Search in summary...", scale=3)
                    filter_type = gr.Dropdown(["All", "Commercial Invoice", "Proforma Invoice", "Bill of Lading", "Air Waybill", "Packing List", "Certificate of Origin", "Letter of Credit"], value="All", label="Doc Type", scale=1)
                with gr.Row():
                    min_amt = gr.Number(label="Min Amount", value=0, scale=1)
                    max_amt = gr.Number(label="Max Amount", value=0, scale=1)
                    date_from = gr.Textbox(label="Date From (YYYY-MM-DD)", placeholder="2024-01-01", scale=1)
                    date_to = gr.Textbox(label="Date To (YYYY-MM-DD)", placeholder="2024-12-31", scale=1)
                with gr.Row(variant="compact"):
                    spec_1 = gr.Textbox(label="Special Filter 1", visible=False, scale=1)
                    spec_2 = gr.Textbox(label="Special Filter 2", visible=False, scale=1)
                    spec_3 = gr.Textbox(label="Special Filter 3", visible=False, scale=1)
                with gr.Row():
                    search_btn = gr.Button("Apply Filters & Search", variant="primary", scale=4)
                    refresh_btn = gr.Button("Reset All", scale=1)
            
            doc_list = gr.Dataframe(headers=["UUID", "Type", "Name", "ID", "Date", "Amount", "Vessel", "POL", "POD", "Incoterms", "Summary"], interactive=False)
            with gr.Row():
                with gr.Column():
                    selected_id = gr.Textbox(label="UUID", interactive=False)
                    selected_summary = gr.Textbox(label="Summary", interactive=False, lines=2)
                with gr.Column():
                    selected_json = gr.Code(label="JSON", language="json")
                    with gr.Row():
                        update_btn = gr.Button("Update")
                        delete_btn = gr.Button("Delete", variant="stop")
                    list_status = gr.Textbox(label="Operation Status", interactive=False, lines=1)

        # --- TAB 3: AI QUERY ---
        with gr.Tab("STEP 3: AI Query (Text-to-NoSQL)"):
            gr.Markdown("### 🤖 Ask AI about your documents")
            gr.Markdown("Ask complex questions like: *'Find invoices from Samsung greater than $5000'* or *'Show me BLs shipped from Busan'*.")
            
            with gr.Row():
                ai_query_input = gr.Textbox(label="Natural Language Query", placeholder="e.g., Find Samsung invoices with amount > 5000", scale=4)
                ai_search_btn = gr.Button("AI Search", variant="primary", scale=1)
            
            ai_status_output = gr.Markdown("Ready")

            with gr.Row():
                with gr.Column(scale=1):
                    ai_filter_view = gr.JSON(label="Generated NoSQL Filter (Debug)")
                with gr.Column(scale=3):
                    ai_results = gr.Dataframe(headers=["UUID", "Type", "Name", "ID", "Date", "Amount", "Vessel", "POL", "POD", "Incoterms", "Summary"], interactive=False)

            ai_search_btn.click(handle_nosql_search, inputs=[ai_query_input], outputs=[ai_results, ai_filter_view, ai_status_output])
            ai_query_input.submit(handle_nosql_search, inputs=[ai_query_input], outputs=[ai_results, ai_filter_view, ai_status_output])

        # --- TAB 4: SETTINGS ---
        with gr.Tab("STEP 4: Settings"):
            gr.Markdown("### 🛠️ System Configuration")
            with gr.Row():
                device_radio = gr.Radio(["Auto", "GPU", "CPU"], value="Auto", label="Device Selection", scale=1)
                low_vram_chk = gr.Checkbox(label="Low VRAM Mode (Faster)", value=False, info="Use 768px resolution for speed", scale=1)
                unload_btn = gr.Button("🧹 Unload Model (Free Memory)", variant="secondary", scale=1)


        # --- TAB 4: HELP ---
        with gr.Tab("STEP 4: Help"):
            gr.Markdown("### ⚙️ System Guide")
            gr.Markdown("""
            - **Device Selection (in Settings):** 
                - **Auto:** Automatically selects GPU if available.
                - **GPU:** Forces GPU usage (NVIDIA/AMD).
                - **CPU:** Forces CPU usage (Slower).
            - **Low VRAM Mode:** Reduces image resolution for faster processing and lower memory usage.
            """)

    gr.Markdown("### 📊 System Status")
    device_status = gr.Textbox(label="Current Status", interactive=False, value="Ready")

    # --- EVENT WIRING ---
    device_radio.change(change_device_handler, inputs=[device_radio], outputs=[device_status])
    unload_btn.click(trigger_unload, outputs=[device_status])

    # Extract Logic with Cancel
    
    def on_extract_start():
        return gr.update(visible=False), gr.update(visible=True)

    def on_extract_end():
        return gr.update(visible=True), gr.update(visible=False)

    extract_event = extract_btn.click(
        fn=on_extract_start,
        outputs=[extract_btn, cancel_btn]
    ).then(
        fn=handle_upload,
        inputs=[img_input, low_vram_chk], 
        outputs=[result_output, device_status]
    )
    
    # Cancel Button
    cancel_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[extract_event]
    ).then(
        fn=on_extract_end, 
        outputs=[extract_btn, cancel_btn]
    )
    
    # Also reset buttons when finished normally
    extract_event.then(
        fn=on_extract_end,
        outputs=[extract_btn, cancel_btn]
    ).then(
        refresh_list, 
        outputs=[doc_list]
    )

    doc_tab.select(refresh_list, outputs=[doc_list])
    
    filter_inputs = [search_query, filter_type, min_amt, max_amt, date_from, date_to, spec_1, spec_2, spec_3]
    
    # Dynamic UI Update
    filter_type.change(handle_type_change, inputs=[filter_type], outputs=[spec_1, spec_2, spec_3])
    
    search_btn.click(handle_search, inputs=filter_inputs, outputs=[doc_list])
    search_query.submit(handle_search, inputs=filter_inputs, outputs=[doc_list])
    refresh_btn.click(refresh_list, outputs=[doc_list])

    doc_list.select(
        load_details, 
        inputs=[doc_list, current_id_state], 
        outputs=[selected_id, selected_json, selected_summary, current_id_state, list_status] 
    ) 
    
    update_btn.click(save_changes, inputs=[selected_id, selected_json], outputs=[list_status]).then(refresh_list, outputs=[doc_list])
    delete_btn.click(delete_record, inputs=[selected_id], outputs=[list_status]).then(refresh_list, outputs=[doc_list])

    demo.load(refresh_list, outputs=[doc_list])

if __name__ == "__main__":
    try:
        demo.launch(server_port=7864)
    except OSError:
        demo.launch()
