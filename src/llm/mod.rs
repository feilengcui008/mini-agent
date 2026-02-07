pub mod claude;
pub mod openai;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[allow(clippy::upper_case_acronyms)]
#[async_trait]
pub trait LLM: Send + Sync {
    async fn complete(&self, messages: &[Message]) -> Result<String>;
}

// Factory to create LLM instances
pub fn create_llm(
    provider: &str,
    model: &str,
    api_key: &str,
    api_url: Option<String>,
) -> Result<Box<dyn LLM>> {
    match provider.to_lowercase().as_str() {
        "openai" => Ok(Box::new(openai::OpenAIClient::new(
            api_key.to_string(),
            model.to_string(),
            api_url,
        ))),
        "claude" => Ok(Box::new(claude::ClaudeClient::new(
            api_key.to_string(),
            model.to_string(),
            api_url,
        ))),
        "minimax" => Ok(Box::new(claude::ClaudeClient::new(
            api_key.to_string(),
            model.to_string(),
            api_url,
        ))),
        _ => Err(anyhow::anyhow!("Unknown provider: {}", provider)),
    }
}
