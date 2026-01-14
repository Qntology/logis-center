use std::time::Instant;

use aha::{
    models::{
        GenerateModel,
        voxcpm::{generate::VoxCPMGenerate, tokenizer::SingleChineseTokenizer},
    },
    utils::audio_utils::{extract_and_save_audio_from_response, save_wav},
};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::{Ok, Result};

#[test]
fn voxcpm1_5_use_message_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda voxcpm1_5_use_message_generate -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/OpenBMB/VoxCPM1.5/";
    let message = r#"
    {
        "model": "voxcpm1.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": 
                        {
                            "url": "https://sis-sample-audio.obs.cn-north-1.myhuaweicloud.com/16k16bit.wav"
                        }
                    },              
                    {
                        "type": "text", 
                        "text": "VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly realistic speech."
                    }
                ]
            }
        ],
        "metadata": {"prompt_text": "华为致力于把数字世界带给每个人，每个家庭，每个组织，构建万物互联的智能世界。"}
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut voxcpm_generate = VoxCPMGenerate::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let generate = voxcpm_generate.generate(mes)?;
    let save_path = extract_and_save_audio_from_response(&generate, "./")?;
    for path in save_path {
        println!("save audio: {}", path);
    }
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}

#[test]
fn voxcpm1_5_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda voxcpm1_5_generate -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/OpenBMB/VoxCPM1.5/";

    let i_start = Instant::now();
    let mut voxcpm_generate = VoxCPMGenerate::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    // let generate = voxcpm_generate.generate_simple("太阳当空照，花儿对我笑，小鸟说早早早".to_string())?;
    let generate = voxcpm_generate.inference(
        "VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly realistic speech.".to_string(),
        Some("啥子小师叔，打狗还要看主人，你再要继续，我就是你的对手".to_string()),
        Some("file://./assets/audio/voice_01.wav".to_string()),
        // Some("一定被灰太狼给吃了，我已经为他准备好了花圈了".to_string()),
        // Some("file://./assets/audio/voice_05.wav".to_string()),
        2,
        4096,
        10,
        2.0,
        // false,
        6.0,
    )?;

    // 创建prompt_cache
    // let _ = voxcpm_generate.build_prompt_cache(
    //     "啥子小师叔，打狗还要看主人，你再要继续，我，就是你的对手".to_string(),
    //     "file://./assets/audio/voice_01.wav".to_string(),
    // )?;
    // // 使用prompt_cache生成语音
    // let generate = voxcpm_generate.generate_use_prompt_cache(
    //     "太阳当空照，花儿对我笑，小鸟说早早早".to_string(),
    //     2,
    //     100,
    //     10,
    //     2.0,
    //     false,
    //     6.0,
    // )?;

    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    save_wav(&generate, "voxcpm1_5.wav", 44100)?;
    Ok(())
}

#[test]
fn voxcpm1_5_tokenizer() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda voxcpm1_5_tokenizer -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/OpenBMB/VoxCPM1.5/";
    let tokenizer = SingleChineseTokenizer::new(model_path)?;
    let ids = tokenizer.encode("你好啊，你吃饭了吗".to_string())?;
    println!("ids: {:?}", ids);
    Ok(())
}
