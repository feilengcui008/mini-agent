use crate::context::ContextManager;
use crate::llm::Message;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
pub struct SessionData {
    pub id: String,
    pub messages: Vec<Message>,
    pub created_at: String,
}

pub struct SessionManager {
    storage_dir: PathBuf,
}

impl SessionManager {
    pub fn new(storage_dir: PathBuf) -> Self {
        if !storage_dir.exists() {
            fs::create_dir_all(&storage_dir).unwrap_or_default();
        }
        Self { storage_dir }
    }

    pub fn save_session(&self, session_id: &str, context: &ContextManager) -> Result<()> {
        let data = SessionData {
            id: session_id.to_string(),
            messages: context.get_history().to_vec(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        let path = self.storage_dir.join(format!("{}.json", session_id));
        let file = fs::File::create(path)?;
        serde_json::to_writer_pretty(file, &data)?;
        Ok(())
    }

    pub fn load_session(&self, session_id: &str, context: &mut ContextManager) -> Result<()> {
        let path = self.storage_dir.join(format!("{}.json", session_id));
        let file = fs::File::open(path)?;
        let data: SessionData = serde_json::from_reader(file)?;

        context.load_history(data.messages);
        Ok(())
    }

    pub fn list_sessions(&self) -> Result<Vec<String>> {
        let mut sessions = Vec::new();
        if self.storage_dir.exists() {
            for entry in fs::read_dir(&self.storage_dir)? {
                let entry = entry?;
                if let Some(name) = entry
                    .file_name()
                    .to_str()
                    .filter(|name| name.ends_with(".json"))
                {
                    sessions.push(name.trim_end_matches(".json").to_string());
                }
            }
        }
        Ok(sessions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;
    use tempfile::tempdir;

    #[test]
    fn test_session_manager() {
        let dir = tempdir().unwrap();
        let manager = SessionManager::new(dir.path().to_path_buf());
        let mut ctx = ContextManager::new(1000);

        ctx.add_message(Message {
            role: Role::User,
            content: "Hello".to_string(),
        });

        manager.save_session("test_session", &ctx).unwrap();

        let mut ctx2 = ContextManager::new(1000);
        manager.load_session("test_session", &mut ctx2).unwrap();

        assert_eq!(ctx2.get_history().len(), 1);
        assert_eq!(ctx2.get_history()[0].content, "Hello");

        let sessions = manager.list_sessions().unwrap();
        assert!(sessions.contains(&"test_session".to_string()));
    }
}
