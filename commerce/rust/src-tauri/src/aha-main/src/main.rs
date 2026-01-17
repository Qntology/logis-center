use std::{net::IpAddr, str::FromStr, time::Duration};

use aha::{models::WhichModel, utils::get_default_save_dir};
use clap::Parser;
use modelscope::ModelScope;
use rocket::{
    Config,
    data::{ByteUnit, Limits},
    routes,
};
use tokio::time::sleep;

use crate::api::init;
mod api;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1")]
    address: String,

    #[arg(short, long, default_value_t = 10100)]
    port: u16,

    #[arg(short, long)]
    model: WhichModel,

    #[arg(long)]
    weight_path: Option<String>,

    #[arg(long)]
    save_dir: Option<String>,

    #[arg(long)]
    download_retries: Option<u32>,
}
async fn download_model(model_id: &str, save_dir: &str, max_retries: u32) -> anyhow::Result<()> {
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        println!(
            "Attempting to download model (attempt {}/{})",
            attempts, max_retries
        );

        match ModelScope::download(model_id, save_dir).await {
            Ok(()) => {
                println!("Model downloaded successfully");
                return Ok(());
            }
            Err(e) => {
                if attempts >= max_retries {
                    return Err(anyhow::anyhow!(
                        "Failed to download model after {} attempts. Last error: {}",
                        max_retries,
                        e
                    ));
                }

                println!(
                    "Download failed (attempt {}): {}. Retrying in 2 seconds...",
                    attempts, e
                );
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let model_id = match &args.model {
        WhichModel::MiniCPM4_0_5B => "OpenBMB/MiniCPM4-0.5B",
        WhichModel::Qwen2_5vl3B => "Qwen/Qwen2.5-VL-3B-Instruct",
        WhichModel::Qwen2_5vl7B => "Qwen/Qwen2.5-VL-7B-Instruct",
        WhichModel::Qwen3_0_6B => "Qwen/Qwen3-0.6B",
        WhichModel::Qwen3vl2B => "Qwen/Qwen3-VL-2B-Instruct",
        WhichModel::Qwen3vl4B => "Qwen/Qwen3-VL-4B-Instruct",
        WhichModel::Qwen3vl8B => "Qwen/Qwen3-VL-8B-Instruct",
        WhichModel::Qwen3vl32B => "Qwen/Qwen3-VL-32B-Instruct",
        WhichModel::DeepSeekOCR => "deepseek-ai/DeepSeek-OCR",
        WhichModel::HunyuanOCR => "Tencent-Hunyuan/HunyuanOCR",
        WhichModel::PaddleOCRVL => "PaddlePaddle/PaddleOCR-VL",
        WhichModel::RMBG2_0 => "AI-ModelScope/RMBG-2.0",
        WhichModel::VoxCPM => "OpenBMB/VoxCPM-0.5B",
        WhichModel::VoxCPM1_5 => "OpenBMB/VoxCPM1.5",
        WhichModel::GlmASRNano2512 => "ZhipuAI/GLM-ASR-Nano-2512",
        WhichModel::FunASRNano2512 => "FunAudioLLM/Fun-ASR-Nano-2512",
    };
    let model_path = match &args.weight_path {
        Some(path) => path.clone(),
        None => {
            let save_dir = match &args.save_dir {
                Some(dir) => dir.clone(),
                None => get_default_save_dir().expect("Failed to get home directory"),
            };
            let max_retries = args.download_retries.unwrap_or(3);
            download_model(model_id, &save_dir, max_retries).await?;
            save_dir + "/" + model_id
        }
    };
    // println!("-------------------download path: {}", model_path);
    init(args.model, model_path)?;
    start_http_server(&args).await?;

    Ok(())
}

pub(crate) async fn start_http_server(args: &Args) -> anyhow::Result<()> {
    let mut builder = rocket::build().configure(Config {
        address: IpAddr::from_str(&args.address)?,
        port: args.port,
        limits: Limits::default()
            .limit("string", ByteUnit::Mebibyte(5))
            .limit("json", ByteUnit::Mebibyte(5))
            .limit("data-form", ByteUnit::Mebibyte(100))
            .limit("file", ByteUnit::Mebibyte(100)),
        ..Config::default()
    });

    builder = builder.mount("/chat", routes![api::chat]);
    // /images/remove_background
    builder = builder.mount("/images", routes![api::remove_background]);
    // /images/speech
    builder = builder.mount("/audio", routes![api::speech]);

    builder.launch().await?;
    Ok(())
}

// fn main() {
//     println!("Hello, world!");
// }
