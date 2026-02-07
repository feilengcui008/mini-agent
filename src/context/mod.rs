use crate::llm::{LLM, Message, Role};
use anyhow::Result;
use std::sync::Arc;

pub struct ContextManager {
    history: Vec<Message>,
    max_tokens: usize,
    llm: Option<Arc<dyn LLM>>,
}

impl ContextManager {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            history: Vec::new(),
            max_tokens,
            llm: None,
        }
    }

    pub fn set_llm(&mut self, llm: Arc<dyn LLM>) {
        self.llm = Some(llm);
    }

    pub fn add_message(&mut self, message: Message) {
        self.history.push(message);
    }

    pub fn get_history(&self) -> &[Message] {
        &self.history
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn load_history(&mut self, history: Vec<Message>) {
        self.history = history;
    }

    pub fn inject_system_prompt(&mut self, prompt: String) {
        if let Some(Message {
            role: Role::System, ..
        }) = self.history.first()
        {
            self.history[0].content = prompt;
            return;
        }
        self.history.insert(
            0,
            Message {
                role: Role::System,
                content: prompt,
            },
        );
    }

    pub async fn compress(&mut self) -> Result<()> {
        let current_len: usize = self.history.iter().map(|m| m.content.len()).sum();

        // Approx 4 chars per token. If context exceeds limit, compress.
        if current_len / 4 > self.max_tokens {
            if self.history.len() <= 5 {
                return Ok(());
            }

            let system_msg = if let Some(first) = self.history.first() {
                if first.role == Role::System {
                    Some(first.clone())
                } else {
                    None
                }
            } else {
                None
            };

            let last_n = 4;
            let start_idx = if system_msg.is_some() { 1 } else { 0 };
            let end_idx = self.history.len().saturating_sub(last_n);

            if start_idx >= end_idx {
                return Ok(());
            }

            let to_summarize = &self.history[start_idx..end_idx];
            let summary_content = to_summarize
                .iter()
                .map(|m| format!("{:?}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n");

            // If we have an LLM, use it to summarize. Otherwise just truncate with a placeholder.
            let summary = if let Some(llm) = &self.llm {
                let prompt = format!(
                    "Summarize the following conversation history into a single paragraph. Ignore system messages if any.\n\n{}",
                    summary_content
                );
                // We use a separate ephemeral request for summary
                match llm
                    .complete(&[Message {
                        role: Role::User,
                        content: prompt,
                    }])
                    .await
                {
                    Ok(s) => s,
                    Err(_) => "... Conversation compressed (summary failed) ...".to_string(),
                }
            } else {
                "... Old conversation compressed ...".to_string()
            };

            let summary_msg = Message {
                role: Role::System,
                content: format!("Previous conversation summary: {}", summary),
            };

            let mut new_history = Vec::new();
            if let Some(sys) = system_msg {
                new_history.push(sys);
            }
            new_history.push(summary_msg);
            new_history.extend_from_slice(&self.history[end_idx..]);

            self.history = new_history;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_context_compression() {
        let mut ctx = ContextManager::new(10); // Very small limit to trigger compression

        ctx.add_message(Message {
            role: Role::System,
            content: "You are a bot".to_string(),
        });
        for i in 0..10 {
            ctx.add_message(Message {
                role: Role::User,
                content: format!("Message {}", i),
            });
            ctx.add_message(Message {
                role: Role::Assistant,
                content: format!("Reply {}", i),
            });
        }

        assert_eq!(ctx.get_history().len(), 21);

        ctx.compress().await.unwrap();

        // Should have System + Summary + Last 4
        // Or System + Summary + ...
        // Logic: System(1) + Summary(1) + Last 4 messages = 6 messages
        assert!(ctx.get_history().len() <= 10);
        // 1 System + 1 Summary + 4 last = 6.
        assert_eq!(ctx.get_history().len(), 6);
        assert!(ctx.get_history()[1].content.contains("summary"));
    }
}
