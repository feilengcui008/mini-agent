use super::{LLM, Message, Role};
use anyhow::{Context, Result};
use async_trait::async_trait;
use log::{debug, error};
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct ClaudeClient {
    client: Client,
    api_key: String,
    model: String,
    api_url: String,
}

impl ClaudeClient {
    pub fn new(api_key: String, model: String, api_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            api_url: api_url.unwrap_or_else(|| "https://api.anthropic.com/v1/messages".to_string()),
        }
    }
}

#[derive(Serialize, Debug)]
struct ClaudeRequest {
    model: String,
    messages: Vec<ClaudeMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    stream: bool,
}

#[derive(Serialize, Clone, Debug)]
struct ClaudeMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        #[serde(default)]
        _index: usize,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        #[serde(default)]
        _index: usize,
        delta: ContentBlockDelta,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ContentBlockDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
}

#[async_trait]
impl LLM for ClaudeClient {
    async fn complete(&self, messages: &[Message]) -> Result<String> {
        let mut claude_messages = Vec::new();
        let mut system_prompt = None;

        for m in messages {
            match m.role {
                Role::System => system_prompt = Some(m.content.clone()),
                Role::User => claude_messages.push(ClaudeMessage {
                    role: "user".to_string(),
                    content: m.content.clone(),
                }),
                Role::Assistant => claude_messages.push(ClaudeMessage {
                    role: "assistant".to_string(),
                    content: m.content.clone(),
                }),
            }
        }

        let request_body = ClaudeRequest {
            model: self.model.clone(),
            messages: claude_messages,
            max_tokens: 4096, // Default max tokens
            system: system_prompt,
            stream: true,
        };

        //debug!(
        //    "Sending request to Claude compatable api, model: {}, api_url: {}, request body: {:?}",
        //    self.model, self.api_url, request_body
        //);

        let res = self
            .client
            .post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .context("Failed to send request to Claude")?;

        let status = res.status();
        if !status.is_success() {
            let error_text = res.text().await?;
            error!("Claude API error. Status: {}, Body: {}", status, error_text);
            return Err(anyhow::anyhow!("Claude API error: {}", error_text));
        }

        let body_text = res
            .text()
            .await
            .context("Failed to read Claude response text")?;

        let mut thinking = String::new();
        let mut text = String::new();

        for line in body_text.lines() {
            if let Some(json_str) = line.strip_prefix("data: ") {
                if json_str.trim() == "[DONE]" {
                    break;
                }
                // Parse SSE event
                match serde_json::from_str::<StreamEvent>(json_str) {
                    Ok(event) => {
                        match event {
                            StreamEvent::ContentBlockStart { content_block, .. } => {
                                match content_block {
                                    ContentBlock::Thinking { thinking: t } => thinking.push_str(&t),
                                    ContentBlock::Text { text: t } => text.push_str(&t),
                                }
                            }
                            StreamEvent::ContentBlockDelta { delta, .. } => match delta {
                                ContentBlockDelta::ThinkingDelta { thinking: t } => {
                                    thinking.push_str(&t)
                                }
                                ContentBlockDelta::TextDelta { text: t } => text.push_str(&t),
                            },
                            _ => {} // Ignore other events
                        }
                    }
                    Err(e) => {
                        debug!("Failed to parse SSE event: {}. Json: {}", e, json_str);
                    }
                }
            }
        }

        debug!(
            "Claude compatable api final response - Thinking: {}, Text: {}",
            thinking, text
        );

        if !thinking.is_empty() {
            Ok(format!("<thinking>\n{}\n</thinking>\n{}", thinking, text))
        } else {
            Ok(text)
        }
    }
}
