//! Read-only ONNX inspector for OxiBonsai.
//!
//! Parses an `.onnx` file via `oxionnx-proto` and prints a structured summary
//! (metadata, op histogram, initializer shapes/dtypes, DequantizeLinear wiring,
//! Qwen3 name-pattern matches, graph I/O). Never touches external weight files.
//!
//! Usage: cargo run -p oxibonsai-model --example onnx_inspect -- <path/to/model.onnx>

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use oxionnx_proto::parser::parse_model;
use oxionnx_proto::types::{AttributeProto, ModelProto, NodeProto, TensorProto};

fn main() {
    if let Err(e) = run() {
        eprintln!("onnx_inspect error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().ok_or_else(|| {
        "usage: onnx_inspect <path/to/model.onnx> [init_detail_limit]".to_string()
    })?;
    let init_limit: usize = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(40);

    let bytes = std::fs::read(&path).map_err(|e| format!("failed to read {path}: {e}"))?;
    let model = parse_model(&bytes)?;

    print_metadata(&model, &path, bytes.len());
    print_op_histogram(&model);
    print_initializer_summary(&model);
    print_initializer_detail(&model, init_limit);
    print_dequantize_analysis(&model);
    if let Err(e) = print_matmul_nbits_detail(&model) {
        eprintln!("warning: MatMulNBits section failed: {e}");
    }
    if let Err(e) = print_gqa_detail(&model) {
        eprintln!("warning: GQA section failed: {e}");
    }
    if let Err(e) = analyze_matmul_nbits_weights(&model, &path) {
        eprintln!("warning: MatMulNBits byte-sample section failed: {e}");
    }
    if let Err(e) = print_gather_block_quantized(&model) {
        eprintln!("warning: GatherBlockQuantized section failed: {e}");
    }
    if let Err(e) = print_embed_weight_probe(&model, &path) {
        eprintln!("warning: embed-weight probe section failed: {e}");
    }
    if let Err(e) = print_embed_zp_nibble_check(&model, &path) {
        eprintln!("warning: embed-zp nibble check section failed: {e}");
    }
    if let Err(e) = print_simplified_layernorm(&model) {
        eprintln!("warning: SimplifiedLayerNormalization section failed: {e}");
    }
    print_qwen3_scan(&model);
    print_graph_io(&model);

    Ok(())
}

// ─── dtype helpers ─────────────────────────────────────────────────

fn dtype_name(dt: i32) -> String {
    match dt {
        1 => "float32".into(),
        2 => "uint8".into(),
        3 => "int8".into(),
        4 => "uint16".into(),
        5 => "int16".into(),
        6 => "int32".into(),
        7 => "int64".into(),
        9 => "bool".into(),
        10 => "float16".into(),
        11 => "float64".into(),
        12 => "uint32".into(),
        13 => "uint64".into(),
        14 => "complex64".into(),
        15 => "complex128".into(),
        16 => "bfloat16".into(),
        17 => "float8e4m3fn".into(),
        18 => "float8e4m3fnuz".into(),
        19 => "float8e5m2".into(),
        20 => "float8e5m2fnuz".into(),
        21 => "uint4".into(),
        22 => "int4".into(),
        23 => "float4e2m1".into(),
        n => format!("dtype_{n}"),
    }
}

fn is_external(t: &TensorProto) -> bool {
    t.data_location == 1 || !t.external_data.is_empty()
}

fn external_entry<'a>(t: &'a TensorProto, key: &str) -> Option<&'a str> {
    t.external_data
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

fn external_offset(t: &TensorProto) -> u64 {
    external_entry(t, "offset")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

fn external_length(t: &TensorProto) -> u64 {
    external_entry(t, "length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

fn tensor_size_bytes(t: &TensorProto) -> u64 {
    if is_external(t) {
        external_length(t)
    } else {
        t.raw_data.len() as u64
    }
}

fn loc_name(t: &TensorProto) -> &'static str {
    if is_external(t) {
        "ext"
    } else {
        "inline"
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn fmt_dims(dims: &[i64]) -> String {
    let parts: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(","))
}

fn attr_value_str(attr: &AttributeProto) -> String {
    // ONNX AttributeProto.AttributeType:
    // 1=FLOAT 2=INT 3=STRING 4=TENSOR 6=FLOATS 7=INTS 8=STRINGS
    let v = &attr.value;
    match v.attr_type {
        1 => format!("{}", v.f),
        2 => format!("{}", v.i),
        3 => format!("\"{}\"", truncate(&v.s, 80)),
        4 => match &v.t {
            Some(t) => format!(
                "<tensor dtype={} dims={}>",
                dtype_name(t.data_type),
                fmt_dims(&t.dims)
            ),
            None => "<tensor:none>".into(),
        },
        6 => {
            let parts: Vec<String> = v.floats.iter().map(|x| x.to_string()).collect();
            format!("[{}]", parts.join(","))
        }
        7 => {
            let parts: Vec<String> = v.ints.iter().map(|x| x.to_string()).collect();
            format!("[{}]", parts.join(","))
        }
        8 => {
            let parts: Vec<String> = v
                .strings
                .iter()
                .map(|s| format!("\"{}\"", truncate(s, 40)))
                .collect();
            format!("[{}]", parts.join(","))
        }
        n => format!("<attr_type {n}>"),
    }
}

// ─── section printers ─────────────────────────────────────────────

fn print_metadata(model: &ModelProto, path: &str, file_size: usize) {
    println!("== METADATA ==");
    println!("file: {path}");
    println!("file_size_bytes: {file_size}");
    println!("ir_version: {}", model.ir_version);
    println!("producer_name: {}", model.producer_name);
    println!("producer_version: {}", model.producer_version);
    println!("domain: {}", model.domain);
    println!("model_version: {}", model.model_version);
    println!("graph_name: {}", model.graph.name);

    print!("opset_imports:");
    if model.opset_imports.is_empty() {
        println!(" (none)");
    } else {
        let parts: Vec<String> = model
            .opset_imports
            .iter()
            .map(|o| {
                let dom = if o.domain.is_empty() {
                    "ai.onnx"
                } else {
                    &o.domain
                };
                format!("{}@{}", dom, o.version)
            })
            .collect();
        println!(" {}", parts.join(", "));
    }

    if model.metadata_props.is_empty() {
        println!("metadata_props: (none)");
    } else {
        println!("metadata_props:");
        for (k, v) in &model.metadata_props {
            println!("  {k}: {}", truncate(v, 200));
        }
    }
    println!();
}

fn print_op_histogram(model: &ModelProto) {
    println!("== OP HISTOGRAM ==");
    let mut counts: HashMap<&str, u64> = HashMap::new();
    for n in &model.graph.nodes {
        *counts.entry(n.op_type.as_str()).or_insert(0) += 1;
    }
    let mut rows: Vec<(&str, u64)> = counts.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    println!("total_nodes: {}", model.graph.nodes.len());
    for (op, n) in &rows {
        println!("{:>6}  {}", n, op);
    }
    println!();
}

fn print_initializer_summary(model: &ModelProto) {
    println!("== INITIALIZER SUMMARY ==");
    let inits = &model.graph.initializers;
    let total = inits.len();
    let mut inline_count = 0u64;
    let mut ext_count = 0u64;
    let mut inline_bytes = 0u64;
    let mut ext_bytes = 0u64;
    let mut data_location_set = 0u64;
    let mut by_dtype: HashMap<i32, u64> = HashMap::new();

    for t in inits {
        *by_dtype.entry(t.data_type).or_insert(0) += 1;
        if t.data_location == 1 {
            data_location_set += 1;
        }
        if is_external(t) {
            ext_count += 1;
            ext_bytes += external_length(t);
        } else {
            inline_count += 1;
            inline_bytes += t.raw_data.len() as u64;
        }
    }

    println!("total_initializers: {total}");
    println!("inline_count: {inline_count} (raw_data bytes: {inline_bytes})");
    println!("external_count: {ext_count} (external length bytes: {ext_bytes})");
    println!("data_location_set: {data_location_set} (of {total})");
    println!("total_bytes: {}", inline_bytes + ext_bytes);

    let mut rows: Vec<(i32, u64)> = by_dtype.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("count_by_dtype:");
    for (dt, n) in rows {
        println!("  {} ({}): {}", dt, dtype_name(dt), n);
    }
    println!();
}

fn print_initializer_detail(model: &ModelProto, detail_limit: usize) {
    println!("== INITIALIZER DETAIL (first {detail_limit}) ==");
    let inits = &model.graph.initializers;
    let limit = detail_limit.min(inits.len());
    for t in inits.iter().take(limit) {
        let loc = loc_name(t);
        let name = truncate(&t.name, 100);
        let dn = dtype_name(t.data_type);
        let dims = fmt_dims(&t.dims);
        let size = tensor_size_bytes(t);
        println!(
            "{:<100} {:<10} dims={:<20} loc={:<6} size_bytes={}",
            name, dn, dims, loc, size
        );
    }
    if inits.len() > limit {
        println!("... ({} more initializers)", inits.len() - limit);
    }
    println!();
}

fn find_initializer<'a>(model: &'a ModelProto, name: &str) -> Option<&'a TensorProto> {
    model.graph.initializers.iter().find(|t| t.name == name)
}

fn print_dequantize_analysis(model: &ModelProto) {
    println!("== DEQUANTIZELINEAR ANALYSIS ==");

    let quant_ops = [
        "DequantizeLinear",
        "QuantizeLinear",
        "QLinearMatMul",
        "MatMulNBits",
        "MatMulInteger",
        "ConvInteger",
    ];
    let mut op_counts: HashMap<&str, u64> = HashMap::new();
    for name in &quant_ops {
        op_counts.insert(*name, 0);
    }
    for node in &model.graph.nodes {
        if let Some(c) = op_counts.get_mut(node.op_type.as_str()) {
            *c += 1;
        }
    }
    println!("quant_op_totals:");
    for name in &quant_ops {
        println!("  {}: {}", name, op_counts.get(name).copied().unwrap_or(0));
    }
    println!();

    let dqs: Vec<&NodeProto> = model
        .graph
        .nodes
        .iter()
        .filter(|n| n.op_type == "DequantizeLinear")
        .collect();
    let total_dq = dqs.len();
    let show = 10.min(total_dq);
    println!("first {show} DequantizeLinear nodes (of {total_dq}):");

    for (idx, node) in dqs.iter().take(show).enumerate() {
        let node_name = if node.name.is_empty() {
            "<anon>"
        } else {
            node.name.as_str()
        };
        println!(
            "--- [{idx}] node={} op=DequantizeLinear",
            truncate(node_name, 80)
        );
        let role_names = ["quantized", "scale", "zero_point"];
        for (i, input) in node.inputs.iter().enumerate() {
            let role = role_names.get(i).copied().unwrap_or("extra");
            match find_initializer(model, input) {
                Some(t) => {
                    let loc = loc_name(t);
                    println!(
                        "  in[{i}] ({role}): {} | dtype={} dims={} loc={} size_bytes={}",
                        truncate(input, 60),
                        dtype_name(t.data_type),
                        fmt_dims(&t.dims),
                        loc,
                        tensor_size_bytes(t),
                    );
                }
                None => {
                    println!(
                        "  in[{i}] ({role}): {} | (not an initializer — node output)",
                        truncate(input, 60),
                    );
                }
            }
        }
        for out in &node.outputs {
            println!("  out: {}", truncate(out, 60));
        }
        if node.attributes.is_empty() {
            println!("  attrs: (none)");
        } else {
            for attr in &node.attributes {
                println!("  attr: {}={}", attr.name, attr_value_str(attr));
            }
        }
    }
    if total_dq > show {
        println!(
            "... ({} more DequantizeLinear nodes not shown)",
            total_dq - show
        );
    }
    println!();
}

// ─── Section A: MatMulNBits detail ────────────────────────────────

fn attr_int(attr: &AttributeProto) -> Option<i64> {
    if attr.value.attr_type == 2 {
        Some(attr.value.i)
    } else {
        None
    }
}

fn node_attr<'a>(node: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    node.attributes.iter().find(|a| a.name == name)
}

fn print_matmul_nbits_detail(model: &ModelProto) -> Result<(), String> {
    println!("== MATMULNBITS ANALYSIS ==");
    let nodes: Vec<&NodeProto> = model
        .graph
        .nodes
        .iter()
        .filter(|n| n.op_type == "MatMulNBits")
        .collect();
    let total = nodes.len();
    println!("matmulnbits_count: {total}");

    let show = 3.min(total);
    println!("first {show} MatMulNBits nodes (of {total}):");

    let role_names = [
        "A(activation)",
        "B(weight)",
        "scales",
        "zero_points",
        "g_idx",
        "bias",
    ];

    for (idx, node) in nodes.iter().take(show).enumerate() {
        let node_name = if node.name.is_empty() {
            "<anon>"
        } else {
            node.name.as_str()
        };
        let domain = if node.domain.is_empty() {
            "<default>"
        } else {
            node.domain.as_str()
        };
        println!(
            "--- [{idx}] node={} op=MatMulNBits domain={}",
            truncate(node_name, 80),
            domain
        );
        if node.attributes.is_empty() {
            println!("  attrs: (none)");
        } else {
            for attr in &node.attributes {
                println!("  attr: {}={}", attr.name, attr_value_str(attr));
            }
        }
        for (i, input) in node.inputs.iter().enumerate() {
            let role = role_names.get(i).copied().unwrap_or("extra");
            match find_initializer(model, input) {
                Some(t) => {
                    let loc = loc_name(t);
                    println!(
                        "  in[{i}] ({role}): {} | dtype={} dims={} loc={} size_bytes={}",
                        truncate(input, 60),
                        dtype_name(t.data_type),
                        fmt_dims(&t.dims),
                        loc,
                        tensor_size_bytes(t),
                    );
                }
                None => {
                    println!(
                        "  in[{i}] ({role}): {} | (activation — node output)",
                        truncate(input, 60),
                    );
                }
            }
        }
        for out in &node.outputs {
            println!("  out: {}", truncate(out, 60));
        }
    }

    // Aggregate histogram over (bits, block_size, N, K_rounded_to_128).
    let mut tuple_hist: HashMap<(i64, i64, i64, i64), u64> = HashMap::new();
    for node in &nodes {
        let bits = node_attr(node, "bits").and_then(attr_int).unwrap_or(-1);
        let block_size = node_attr(node, "block_size")
            .and_then(attr_int)
            .unwrap_or(-1);
        let n_attr = node_attr(node, "N").and_then(attr_int).unwrap_or(-1);
        let k_attr = node_attr(node, "K").and_then(attr_int).unwrap_or(-1);
        let k_round = if k_attr >= 0 {
            (k_attr / 128) * 128
        } else {
            -1
        };
        *tuple_hist
            .entry((bits, block_size, n_attr, k_round))
            .or_insert(0) += 1;
    }
    let mut rows: Vec<((i64, i64, i64, i64), u64)> = tuple_hist.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("shape_histogram (bits, block_size, N, K_rounded_to_128) -> count:");
    let hist_show = 20.min(rows.len());
    for ((bits, bs, n_a, k_r), cnt) in rows.iter().take(hist_show) {
        println!(
            "  bits={:<3} block_size={:<5} N={:<8} K_r128={:<8} count={}",
            bits, bs, n_a, k_r, cnt
        );
    }
    if rows.len() > hist_show {
        println!("  ... ({} more tuples)", rows.len() - hist_show);
    }

    // Trace the lm_head B-input back through any Reshape/Transpose nodes to
    // find the initializer it ultimately comes from.
    println!("-- nodes producing *_weight_quant_matmul outputs (trace lm_head B) --");
    for node in &model.graph.nodes {
        for out in &node.outputs {
            if out.contains("weight_quant_matmul") {
                let node_name = if node.name.is_empty() {
                    "<anon>"
                } else {
                    node.name.as_str()
                };
                println!(
                    "  node={} op={} out={}",
                    truncate(node_name, 80),
                    node.op_type,
                    truncate(out, 80)
                );
                for (i, input) in node.inputs.iter().enumerate() {
                    match find_initializer(model, input) {
                        Some(t) => println!(
                            "    in[{i}]: {} | dtype={} dims={} loc={}",
                            truncate(input, 80),
                            dtype_name(t.data_type),
                            fmt_dims(&t.dims),
                            loc_name(t),
                        ),
                        None => println!("    in[{i}]: {} | (activation)", truncate(input, 80),),
                    }
                }
            }
        }
    }

    // Dump the lm_head MatMulNBits node (the one with the largest N — the
    // projection onto vocab size).
    println!("-- lm_head MatMulNBits (largest N) --");
    let lm_head = nodes
        .iter()
        .max_by_key(|n| node_attr(n, "N").and_then(attr_int).unwrap_or(-1));
    if let Some(node) = lm_head {
        let node_name = if node.name.is_empty() {
            "<anon>"
        } else {
            node.name.as_str()
        };
        println!("  node_name: {}", truncate(node_name, 120));
        for attr in &node.attributes {
            println!("  attr: {}={}", attr.name, attr_value_str(attr));
        }
        for (i, input) in node.inputs.iter().enumerate() {
            match find_initializer(model, input) {
                Some(t) => {
                    println!(
                        "  in[{i}]: {} | dtype={} dims={} loc={}",
                        truncate(input, 120),
                        dtype_name(t.data_type),
                        fmt_dims(&t.dims),
                        loc_name(t),
                    );
                }
                None => {
                    println!(
                        "  in[{i}]: {} | (activation — node output)",
                        truncate(input, 120),
                    );
                }
            }
        }
        for out in &node.outputs {
            println!("  out: {}", truncate(out, 120));
        }
    }

    println!();
    Ok(())
}

// ─── Section B: GroupQueryAttention detail ────────────────────────

fn print_gqa_detail(model: &ModelProto) -> Result<(), String> {
    println!("== GQA ANALYSIS ==");
    let nodes: Vec<&NodeProto> = model
        .graph
        .nodes
        .iter()
        .filter(|n| n.op_type == "GroupQueryAttention")
        .collect();
    let total = nodes.len();
    println!("group_query_attention_count: {total}");

    let show = 3.min(total);
    println!("first {show} GroupQueryAttention nodes (of {total}):");

    // Microsoft GQA input order (comment-only — actual name comes from node.inputs[i]).
    let role_names = [
        "query",
        "key",
        "value",
        "past_key",
        "past_value",
        "seqlens_k",
        "total_sequence_length",
        "cos_cache",
        "sin_cache",
    ];

    for (idx, node) in nodes.iter().take(show).enumerate() {
        let node_name = if node.name.is_empty() {
            "<anon>"
        } else {
            node.name.as_str()
        };
        let domain = if node.domain.is_empty() {
            "<default>"
        } else {
            node.domain.as_str()
        };
        println!(
            "--- [{idx}] node={} op=GroupQueryAttention domain={}",
            truncate(node_name, 80),
            domain
        );
        if node.attributes.is_empty() {
            println!("  attrs: (none)");
        } else {
            for attr in &node.attributes {
                println!("  attr: {}={}", attr.name, attr_value_str(attr));
            }
        }
        for (i, input) in node.inputs.iter().enumerate() {
            let role = role_names.get(i).copied().unwrap_or("extra");
            match find_initializer(model, input) {
                Some(t) => {
                    let loc = loc_name(t);
                    println!(
                        "  in[{i}] (/*{role}*/): {} | dtype={} dims={} loc={} size_bytes={}",
                        truncate(input, 60),
                        dtype_name(t.data_type),
                        fmt_dims(&t.dims),
                        loc,
                        tensor_size_bytes(t),
                    );
                }
                None => {
                    println!(
                        "  in[{i}] (/*{role}*/): {} | (activation — node output)",
                        truncate(input, 60),
                    );
                }
            }
        }
        for out in &node.outputs {
            println!("  out: {}", truncate(out, 60));
        }
    }
    println!();
    Ok(())
}

// ─── Section C: MatMulNBits B-tensor byte sample ──────────────────

fn analyze_matmul_nbits_weights(model: &ModelProto, onnx_path: &str) -> Result<(), String> {
    println!("== MATMULNBITS B-TENSOR BYTE SAMPLE ==");

    let node = match model
        .graph
        .nodes
        .iter()
        .find(|n| n.op_type == "MatMulNBits")
    {
        Some(n) => n,
        None => {
            println!("no MatMulNBits nodes present — skipping");
            println!();
            return Ok(());
        }
    };

    let b_name = node
        .inputs
        .get(1)
        .ok_or_else(|| "first MatMulNBits node has fewer than 2 inputs".to_string())?;
    let b_tensor = match find_initializer(model, b_name) {
        Some(t) => t,
        None => {
            println!(
                "first MatMulNBits B-input {} is not an initializer — skipping byte analysis",
                truncate(b_name, 80)
            );
            println!();
            return Ok(());
        }
    };

    if !is_external(b_tensor) {
        println!("first MatMulNBits B-input is not externalized — skipping byte analysis");
        println!();
        return Ok(());
    }

    let rel_location = external_entry(b_tensor, "location")
        .ok_or_else(|| "external_data is populated but missing a 'location' entry".to_string())?;
    let parent = Path::new(onnx_path)
        .parent()
        .ok_or_else(|| format!("cannot derive parent dir from onnx path {onnx_path}"))?;
    let sidecar_path = parent.join(rel_location);

    let offset = external_offset(b_tensor);
    let mut length = external_length(b_tensor);
    if length == 0 {
        // Fallback: derive from dims assuming uint8 (dtype code 2) or int8 (3).
        // For MatMulNBits packed weights, dtype is uint8 and length = product(dims).
        if b_tensor.data_type == 2 || b_tensor.data_type == 3 {
            let mut prod: u64 = 1;
            for d in &b_tensor.dims {
                if *d < 0 {
                    return Err(format!("negative dim {d} in B-tensor"));
                }
                prod = prod.saturating_mul(*d as u64);
            }
            length = prod;
            println!(
                "(length missing in external_data; derived from dims as {length} bytes using dtype={})",
                dtype_name(b_tensor.data_type)
            );
        } else {
            return Err(format!(
                "length missing in external_data and dtype {} not supported for fallback",
                dtype_name(b_tensor.data_type)
            ));
        }
    }

    let length_usize: usize = usize::try_from(length)
        .map_err(|e| format!("length {length} does not fit in usize: {e}"))?;

    println!("path: {}", sidecar_path.display());
    println!("offset: {offset}");
    println!("length_bytes: {length}");
    println!("length_mb: {:.4}", length as f64 / 1_048_576.0);
    println!("dims: {}", fmt_dims(&b_tensor.dims));
    println!(
        "dtype: {} ({})",
        b_tensor.data_type,
        dtype_name(b_tensor.data_type)
    );
    println!("tensor_name: {}", truncate(&b_tensor.name, 120));

    let mut file = File::open(&sidecar_path)
        .map_err(|e| format!("failed to open sidecar {}: {e}", sidecar_path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("failed to seek to offset {offset}: {e}"))?;
    let mut buf = vec![0u8; length_usize];
    file.read_exact(&mut buf)
        .map_err(|e| format!("failed to read {length} bytes: {e}"))?;

    // Unpack as 2-bit codes — 4 codes per byte, LSB-first.
    let mut hist = [0u64; 4];
    for b in &buf {
        hist[(b & 0b11) as usize] += 1;
        hist[((b >> 2) & 0b11) as usize] += 1;
        hist[((b >> 4) & 0b11) as usize] += 1;
        hist[((b >> 6) & 0b11) as usize] += 1;
    }
    let total_codes: u64 = hist.iter().sum();
    println!(
        "code_histogram: [{}, {}, {}, {}] (total_codes: {})",
        hist[0], hist[1], hist[2], hist[3], total_codes
    );
    let ratio = |h: u64| -> f64 {
        if total_codes == 0 {
            0.0
        } else {
            h as f64 / total_codes as f64
        }
    };
    println!(
        "code_ratios: [{:.4}, {:.4}, {:.4}, {:.4}]",
        ratio(hist[0]),
        ratio(hist[1]),
        ratio(hist[2]),
        ratio(hist[3])
    );
    println!("# interpretation: if h0 ≈ h2 and h1 ≈ 2·h0 (i.e. code `01` is ~50%),");
    println!("#   the scheme is ternary {{-1,0,+1}} with code `01` == 0 (most common");
    println!("#   since LLM weights are sparse around zero). OxiBonsai's TQ2_0_g128");
    println!("#   uses 0b00→-1, 0b01→0, 0b10→+1, 0b11→0 — transformers.js may differ.");
    println!(
        "#   If the 4 codes are more evenly distributed, it's likely symmetric int2 {{-2,-1,0,1}}."
    );

    // First 32 bytes hex.
    let take = 32.min(buf.len());
    let hex: Vec<String> = buf
        .iter()
        .take(take)
        .map(|b| format!("{:02x}", b))
        .collect();
    println!("first_{take}_bytes_hex: {}", hex.join(" "));
    println!();

    Ok(())
}

fn print_gather_block_quantized(model: &ModelProto) -> Result<(), String> {
    println!("== GATHERBLOCKQUANTIZED ANALYSIS ==");
    let nodes: Vec<&NodeProto> = model
        .graph
        .nodes
        .iter()
        .filter(|n| n.op_type == "GatherBlockQuantized")
        .collect();
    let total = nodes.len();
    println!("gather_block_quantized_count: {total}");

    let show = 2.min(total);
    for (idx, node) in nodes.iter().take(show).enumerate() {
        let node_name = if node.name.is_empty() {
            "<anon>"
        } else {
            node.name.as_str()
        };
        let domain = if node.domain.is_empty() {
            "<default>"
        } else {
            node.domain.as_str()
        };
        println!(
            "--- [{idx}] node={} op=GatherBlockQuantized domain={}",
            truncate(node_name, 100),
            domain
        );
        if node.attributes.is_empty() {
            println!("  attrs: (none)");
        } else {
            for attr in &node.attributes {
                println!("  attr: {}={}", attr.name, attr_value_str(attr));
            }
        }
        for (i, input) in node.inputs.iter().enumerate() {
            match find_initializer(model, input) {
                Some(t) => {
                    let loc = loc_name(t);
                    println!(
                        "  in[{i}]: {} | dtype={} dims={} loc={} size_bytes={}",
                        truncate(input, 100),
                        dtype_name(t.data_type),
                        fmt_dims(&t.dims),
                        loc,
                        tensor_size_bytes(t),
                    );
                }
                None => {
                    println!(
                        "  in[{i}]: {} | (activation — node output)",
                        truncate(input, 100),
                    );
                }
            }
        }
        for out in &node.outputs {
            println!("  out: {}", truncate(out, 100));
        }
    }
    println!();
    Ok(())
}

// ─── Embed-weight probe (bits contradiction resolver) ────────────
//
// Resolves the contradiction between GatherBlockQuantized `bits=4` attribute
// and tensor dims that math-wise only fit bits=2. Directly reads the
// TensorProto for `model_embed_tokens_weight_quant`, prints dims/length, and
// computes `bits = 8 * (length / dims[0]) / K`. Also dumps 2-bit and 4-bit
// nibble histograms over the first ~262 KiB of the weight sidecar.

fn read_sidecar_bytes(
    tensor: &TensorProto,
    onnx_path: &str,
    max_bytes: usize,
) -> Result<Vec<u8>, String> {
    let rel_location = external_entry(tensor, "location")
        .ok_or_else(|| "tensor is not externalized (no 'location')".to_string())?;
    let parent = Path::new(onnx_path)
        .parent()
        .ok_or_else(|| format!("cannot derive parent dir from {onnx_path}"))?;
    let sidecar_path = parent.join(rel_location);
    let offset = external_offset(tensor);
    let length = external_length(tensor);
    let want = (length as usize).min(max_bytes);

    let mut file = File::open(&sidecar_path)
        .map_err(|e| format!("open {} failed: {e}", sidecar_path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("seek to {offset} failed: {e}"))?;
    let mut buf = vec![0u8; want];
    file.read_exact(&mut buf)
        .map_err(|e| format!("read {want} bytes failed: {e}"))?;
    Ok(buf)
}

fn derive_k_hidden_size(model: &ModelProto) -> Option<i64> {
    // Prefer: MatMulNBits attribute `K` on any node (all attention/MLP
    // projections use the model hidden dim on one axis). The minimum K is a
    // good proxy for hidden_size because MLP intermediate K values are
    // larger, but head-dim projections may be smaller. Use the most common
    // K instead.
    let mut counts: HashMap<i64, u64> = HashMap::new();
    for node in &model.graph.nodes {
        if node.op_type != "MatMulNBits" {
            continue;
        }
        if let Some(k) = node_attr(node, "K").and_then(attr_int) {
            *counts.entry(k).or_insert(0) += 1;
        }
    }
    // Pick the most common K — this is typically hidden_size since q/k/v/o
    // all share that axis.
    counts.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k)
}

fn print_embed_weight_probe(model: &ModelProto, onnx_path: &str) -> Result<(), String> {
    println!("== EMBED-WEIGHT PROBE (bits contradiction resolver) ==");

    let embed_name = "model_embed_tokens_weight_quant";
    let embed = match find_initializer(model, embed_name) {
        Some(t) => t,
        None => {
            println!("initializer {embed_name} not found — skipping probe");
            println!();
            return Ok(());
        }
    };

    // ── Section A: raw TensorProto facts ─────────────────────────
    println!("-- Section A: TensorProto facts --");
    println!("  name: {}", embed.name);
    println!(
        "  data_type: {} ({})",
        embed.data_type,
        dtype_name(embed.data_type)
    );
    println!("  dims (from tensor.dims): {}", fmt_dims(&embed.dims));
    let location = external_entry(embed, "location").unwrap_or("<missing>");
    let offset = external_offset(embed);
    let length = external_length(embed);
    println!("  external.location: {location}");
    println!("  external.offset:   {offset}");
    println!("  external.length:   {length} bytes");

    let k_hidden = derive_k_hidden_size(model).unwrap_or(-1);
    println!("  derived_K (most-common MatMulNBits K attr): {k_hidden}");

    if let Some(rows) = embed.dims.first().copied() {
        if rows > 0 && length > 0 {
            let bytes_per_row = length / rows as u64;
            println!("  bytes_per_row = length / dims[0] = {length} / {rows} = {bytes_per_row}");
            if k_hidden > 0 {
                // bits = 8 * bytes_per_row / K
                let numerator = 8u64.saturating_mul(bytes_per_row);
                let bits_f = numerator as f64 / k_hidden as f64;
                let bits_int = numerator / (k_hidden as u64);
                let bits_rem = numerator % (k_hidden as u64);
                println!(
                    "  bits = 8 * bytes_per_row / K = {numerator} / {k_hidden} = {bits_f} (int={bits_int}, rem={bits_rem})"
                );
            } else {
                println!("  bits: (cannot compute — K unknown)");
            }
        }
    }

    // ── Section B: 2-bit and 4-bit nibble histograms ─────────────
    println!("-- Section B: code histograms (first 262144 bytes) --");
    let sample = match read_sidecar_bytes(embed, onnx_path, 262_144) {
        Ok(b) => b,
        Err(e) => {
            println!("  sidecar read failed: {e}");
            println!();
            return Ok(());
        }
    };
    println!("  bytes_read: {}", sample.len());

    // 2-bit histogram: 4 buckets over codes 0..3, 4 codes per byte.
    let mut hist2 = [0u64; 4];
    for b in &sample {
        hist2[(b & 0b11) as usize] += 1;
        hist2[((b >> 2) & 0b11) as usize] += 1;
        hist2[((b >> 4) & 0b11) as usize] += 1;
        hist2[((b >> 6) & 0b11) as usize] += 1;
    }
    let total2: u64 = hist2.iter().sum();

    // 4-bit histogram: 16 buckets over codes 0..15, 2 nibbles per byte.
    let mut hist4 = [0u64; 16];
    for b in &sample {
        hist4[(b & 0b1111) as usize] += 1;
        hist4[((b >> 4) & 0b1111) as usize] += 1;
    }
    let total4: u64 = hist4.iter().sum();

    println!("  2-bit histogram (codes 0..3), total_codes={total2}:");
    for (i, h) in hist2.iter().enumerate() {
        let ratio = if total2 == 0 {
            0.0
        } else {
            *h as f64 / total2 as f64
        };
        println!(
            "    code={:>2} ({:02b}) count={:>10}  ratio={:.4}",
            i, i, h, ratio
        );
    }

    println!("  4-bit histogram (codes 0..15), total_codes={total4}:");
    let mut nonzero_buckets = 0u64;
    for (i, h) in hist4.iter().enumerate() {
        let ratio = if total4 == 0 {
            0.0
        } else {
            *h as f64 / total4 as f64
        };
        if *h > 0 {
            nonzero_buckets += 1;
        }
        println!(
            "    code={:>2} ({:04b}) count={:>10}  ratio={:.4}",
            i, i, h, ratio
        );
    }
    println!("  nonzero_4bit_buckets: {nonzero_buckets} / 16");
    println!("  # interpretation: if all 16 buckets are well-populated -> genuine 4-bit;");
    println!(
        "  #                 if only a handful are populated (clumpy) -> 2-bit packed in bytes."
    );

    // ── Section C: compare to lm_head (8B only, may be absent) ───
    println!("-- Section C: embed vs lm_head byte compare --");
    let lm_head_name = "lm_head_MatMul_weight_quant";
    match find_initializer(model, lm_head_name) {
        Some(lm) => {
            let embed_head = match read_sidecar_bytes(embed, onnx_path, 64) {
                Ok(b) => b,
                Err(e) => {
                    println!("  embed first-64 read failed: {e}");
                    println!();
                    return Ok(());
                }
            };
            let lm_head_bytes = match read_sidecar_bytes(lm, onnx_path, 64) {
                Ok(b) => b,
                Err(e) => {
                    println!("  lm_head first-64 read failed: {e}");
                    println!();
                    return Ok(());
                }
            };
            let fmt_hex = |buf: &[u8]| -> String {
                buf.iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            println!("  embed[0..64]:   {}", fmt_hex(&embed_head));
            println!("  lm_head[0..64]: {}", fmt_hex(&lm_head_bytes));
            let identical = embed_head == lm_head_bytes;
            println!("  identical_first_64: {identical}");
            if identical {
                println!(
                    "  -> tied weights (embedding reused for lm_head) — confirmed by byte match"
                );
            } else {
                println!("  -> NOT tied at the byte level (independent initializers)");
            }
        }
        None => {
            println!("  {lm_head_name} not present — skipping (likely tied embeddings)");
        }
    }

    println!();
    Ok(())
}

// ─── Embed-ZP nibble sanity check (bits=4 vs 2-bit-repack) ────────
//
// Verifies that every 4-bit nibble in `model_embed_tokens_weight_zp_4b` is
// ≤ 3. If so, the declared `bits=4` on GatherBlockQuantized is a 2-bit
// zero-point padded into a 4-bit nibble (safe to re-pack). If any nibble
// ≥ 4 is observed, the 2-bit interpretation would silently corrupt the
// embedding dequant path.

fn print_embed_zp_nibble_check(model: &ModelProto, onnx_path: &str) -> Result<(), String> {
    println!("== EMBED-ZP NIBBLE CHECK (bits=4 vs 2-bit re-pack) ==");

    let zp_name = "model_embed_tokens_weight_zp_4b";
    let zp = match find_initializer(model, zp_name) {
        Some(t) => t,
        None => {
            println!("(absent — probably 1.7B model)");
            println!();
            return Ok(());
        }
    };

    println!("  name: {}", zp.name);
    println!(
        "  data_type: {} ({})",
        zp.data_type,
        dtype_name(zp.data_type)
    );
    println!("  dims: {}", fmt_dims(&zp.dims));
    let length = external_length(zp);
    println!("  external.length: {length} bytes");

    // Cap at 4 MiB to avoid OOM; 151669*16 ≈ 2.43 MiB fits in full.
    let bytes = read_sidecar_bytes(zp, onnx_path, 4 * 1024 * 1024)?;
    println!("  bytes_read: {}", bytes.len());

    let mut hist = [0u64; 16];
    for b in &bytes {
        hist[(b & 0x0F) as usize] += 1;
        hist[((b >> 4) & 0x0F) as usize] += 1;
    }
    let total: u64 = hist.iter().sum();

    let mut max_nibble: u8 = 0;
    let mut ge4_count: u64 = 0;
    for (i, h) in hist.iter().enumerate() {
        if *h > 0 && (i as u8) > max_nibble {
            max_nibble = i as u8;
        }
        if i >= 4 {
            ge4_count += *h;
        }
    }

    println!("  total_nibbles: {total}");
    println!("  max_observed_nibble: {max_nibble}");
    println!("  nibbles_ge_4: {ge4_count}");
    println!("  4-bit histogram (codes 0..15):");
    for (i, h) in hist.iter().enumerate() {
        let ratio = if total == 0 {
            0.0
        } else {
            *h as f64 / total as f64
        };
        println!(
            "    code={:>2} ({:04b}) count={:>10}  ratio={:.6}",
            i, i, h, ratio
        );
    }

    if ge4_count == 0 {
        println!("  OK: all nibbles <= 3 -- 2-bit re-pack is valid");
    } else {
        println!(
            "  FAIL: nibbles >= 4 detected (N={ge4_count}) -- bits=4 is genuine, do NOT re-pack"
        );
    }
    println!();
    Ok(())
}

fn print_simplified_layernorm(model: &ModelProto) -> Result<(), String> {
    println!("== SIMPLIFIEDLAYERNORMALIZATION NODES ==");
    let nodes: Vec<&NodeProto> = model
        .graph
        .nodes
        .iter()
        .filter(|n| {
            n.op_type == "SimplifiedLayerNormalization"
                || n.op_type == "SkipSimplifiedLayerNormalization"
        })
        .collect();
    let total = nodes.len();
    println!("total_simplified_layernorm_nodes: {total}");

    // Show first 3 of each flavour.
    for flavour in &[
        "SimplifiedLayerNormalization",
        "SkipSimplifiedLayerNormalization",
    ] {
        let mut printed = 0usize;
        println!("-- first 3 {flavour} nodes --");
        for node in &nodes {
            if node.op_type != *flavour {
                continue;
            }
            if printed >= 3 {
                break;
            }
            let node_name = if node.name.is_empty() {
                "<anon>"
            } else {
                node.name.as_str()
            };
            println!("  [{}] node={}", printed, truncate(node_name, 100),);
            if node.attributes.is_empty() {
                println!("    attrs: (none)");
            } else {
                for attr in &node.attributes {
                    println!("    attr: {}={}", attr.name, attr_value_str(attr));
                }
            }
            for (i, input) in node.inputs.iter().enumerate() {
                match find_initializer(model, input) {
                    Some(t) => {
                        let loc = loc_name(t);
                        println!(
                            "    in[{i}]: {} | dtype={} dims={} loc={} size_bytes={}",
                            truncate(input, 100),
                            dtype_name(t.data_type),
                            fmt_dims(&t.dims),
                            loc,
                            tensor_size_bytes(t),
                        );
                    }
                    None => {
                        println!(
                            "    in[{i}]: {} | (activation — node output)",
                            truncate(input, 100),
                        );
                    }
                }
            }
            for out in &node.outputs {
                println!("    out: {}", truncate(out, 100));
            }
            printed += 1;
        }
    }

    // Find the LAST SkipSimplifiedLayerNormalization before the lm_head
    // MatMulNBits — its scale initializer (input[2]) is the final
    // `model.norm.weight`.
    println!("-- final SkipSimplifiedLayerNormalization (→ output_norm.weight) --");
    let mut last_skip: Option<&NodeProto> = None;
    for node in &model.graph.nodes {
        if node.op_type == "SkipSimplifiedLayerNormalization" {
            last_skip = Some(node);
        }
    }
    if let Some(node) = last_skip {
        let node_name = if node.name.is_empty() {
            "<anon>"
        } else {
            node.name.as_str()
        };
        println!("  last_skip_node: {}", truncate(node_name, 100));
        for (i, input) in node.inputs.iter().enumerate() {
            if let Some(t) = find_initializer(model, input) {
                println!(
                    "    in[{i}]: {} | dtype={} dims={} loc={}",
                    truncate(input, 100),
                    dtype_name(t.data_type),
                    fmt_dims(&t.dims),
                    loc_name(t),
                );
            }
        }
    } else {
        println!("  (no SkipSimplifiedLayerNormalization nodes found)");
    }
    println!();
    Ok(())
}

fn print_qwen3_scan(model: &ModelProto) {
    println!("== QWEN3 NAME PATTERN SCAN ==");
    let patterns: &[(&str, &str)] = &[
        ("embed_tokens.weight", "embed_tokens.weight"),
        ("self_attn.q_proj.weight", "self_attn.q_proj.weight"),
        ("self_attn.k_proj.weight", "self_attn.k_proj.weight"),
        ("self_attn.v_proj.weight", "self_attn.v_proj.weight"),
        ("self_attn.o_proj.weight", "self_attn.o_proj.weight"),
        ("self_attn.q_norm.weight", "self_attn.q_norm.weight"),
        ("self_attn.k_norm.weight", "self_attn.k_norm.weight"),
        ("mlp.gate_proj.weight", "mlp.gate_proj.weight"),
        ("mlp.up_proj.weight", "mlp.up_proj.weight"),
        ("mlp.down_proj.weight", "mlp.down_proj.weight"),
        ("input_layernorm.weight", "input_layernorm.weight"),
        (
            "post_attention_layernorm.weight",
            "post_attention_layernorm.weight",
        ),
        ("model.norm.weight", "model.norm.weight"),
        ("lm_head.weight", "lm_head.weight"),
    ];

    let mut matched: HashMap<usize, u64> = HashMap::new();
    let mut claimed: Vec<bool> = vec![false; model.graph.initializers.len()];

    for (idx, t) in model.graph.initializers.iter().enumerate() {
        for (pi, (_label, needle)) in patterns.iter().enumerate() {
            if t.name.contains(needle) {
                *matched.entry(pi).or_insert(0) += 1;
                claimed[idx] = true;
                break;
            }
        }
    }

    for (pi, (label, _needle)) in patterns.iter().enumerate() {
        println!("  {:<40} {}", label, matched.get(&pi).copied().unwrap_or(0));
    }

    let others: Vec<&TensorProto> = model
        .graph
        .initializers
        .iter()
        .enumerate()
        .filter(|(i, _)| !claimed[*i])
        .map(|(_, t)| t)
        .collect();
    println!("  {:<40} {}", "OTHER (unmatched)", others.len());
    let sample = 30.min(others.len());
    if sample > 0 {
        println!("  OTHER examples (up to {sample}):");
        for t in others.iter().take(sample) {
            println!(
                "    {} [dtype={} dims={}]",
                truncate(&t.name, 80),
                dtype_name(t.data_type),
                fmt_dims(&t.dims),
            );
        }
        if others.len() > sample {
            println!("    ... ({} more)", others.len() - sample);
        }
    }
    println!();
}

fn print_graph_io(model: &ModelProto) {
    println!("== GRAPH I/O ==");
    println!("graph.inputs ({}):", model.graph.inputs.len());
    for n in &model.graph.inputs {
        println!("  {n}");
    }
    println!("graph.outputs ({}):", model.graph.outputs.len());
    for n in &model.graph.outputs {
        println!("  {n}");
    }
}
