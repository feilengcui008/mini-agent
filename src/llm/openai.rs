use super::{LLM, Message, Role};
use anyhow::{Context, Result};
use async_trait::async_trait;
use log::{debug, error};
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OpenAIClient {
    client: Client,
    api_key: String,
    model: String,
    api_url: String,
}

impl OpenAIClient {
    pub fn new(api_key: String, model: String, api_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            api_url: api_url
                .unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".to_string()),
        }
    }
}

#[derive(Serialize, Debug)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
}

#[derive(Serialize, Clone, Debug)]
struct OpenAIMessage {
    role: String,
    content: String,
}

impl From<&Message> for OpenAIMessage {
    fn from(m: &Message) -> Self {
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        };
        Self {
            role: role.to_string(),
            content: m.content.clone(),
        }
    }
}

#[derive(Deserialize, Debug)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize, Debug)]
struct OpenAIChoice {
    message: OpenAIMessageContent,
}

#[derive(Deserialize, Debug)]
struct OpenAIMessageContent {
    content: String,
}

#[async_trait]
impl LLM for OpenAIClient {
    async fn complete(&self, messages: &[Message]) -> Result<String> {
        let req_messages: Vec<OpenAIMessage> = messages.iter().map(|m| m.into()).collect();
        let request_body = OpenAIChatRequest {
            model: self.model.clone(),
            messages: req_messages,
        };

        debug!(
            "Sending request to OpenAI compatible api, model: {}, api_url: {}, request body: {:?}",
            self.model, self.api_url, request_body
        );

        let res = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request_body)
            .send()
            .await
            .context("Failed to send request to OpenAI")?;

        let status = res.status();
        if !status.is_success() {
            let error_text = res.text().await?;
            error!("OpenAI API error. Status: {}, Body: {}", status, error_text);
            return Err(anyhow::anyhow!("OpenAI API error: {}", error_text));
        }

        let response_body: OpenAIChatResponse = res
            .json()
            .await
            .context("Failed to parse OpenAI response")?;
        debug!("OpenAI compatible api response body: {:?}", response_body);

        response_body
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| {
                error!("No choices in OpenAI response");
                anyhow::anyhow!("No choices in OpenAI response")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dotenv::dotenv;
    use std::env;

    #[tokio::test]
    #[ignore] // Skip this test in CI/CD as it requires a real API key
    async fn test_openai_complete() {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let client = OpenAIClient::new(api_key, "gpt-3.5-turbo".to_string(), None);

        let messages = vec![Message {
            role: Role::User,
            content: "Hello, say 'test passed'".to_string(),
        }];

        let result = client.complete(&messages).await;
        assert!(result.is_ok());
        let content = result.unwrap();
        assert!(content.to_lowercase().contains("passed"));
    }
}
