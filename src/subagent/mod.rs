use crate::context::ContextManager;
use crate::llm::LLM;
use crate::tool::ToolRegistry;
use anyhow::Result;
use log::{debug, info, warn};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::watch;
use uuid::Uuid;

// Tool call structure
pub struct ToolCall {
    pub name: String,
    pub args: Value,
}

// SubAgent config
#[derive(Debug, Clone)]
pub struct SubAgentConfig {
    // Task description
    pub task: String,
    // SubAgent type (dynamic or fixed types like code, test, doc)
    pub agent_type: String,
    // Max loop iterations
    pub max_loops: usize,
}

impl SubAgentConfig {
    pub fn new(task: String, agent_type: String, max_loops: usize) -> Self {
        Self {
            task,
            agent_type,
            max_loops,
        }
    }
}

// SubAgent status
#[derive(Debug, Clone, PartialEq)]
pub enum SubAgentStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
}

// SubAgent core struct
pub struct SubAgent {
    // Unique identifier
    pub id: String,
    // Task description
    pub task: String,
    // SubAgent type
    pub agent_type: String,
    // Isolated context
    pub context: ContextManager,
    // Status
    pub status: SubAgentStatus,
    // Result
    pub result: Option<String>,
    // Max loop iterations
    pub max_loops: usize,
}

impl SubAgent {
    pub fn new(config: SubAgentConfig) -> Self {
        let id = Uuid::new_v4()
            .to_string()
            .split('-')
            .next()
            .unwrap_or("")
            .to_string();
        Self {
            id: id.clone(),
            task: config.task,
            agent_type: config.agent_type,
            context: ContextManager::new(8192), // Default max_tokens
            status: SubAgentStatus::Pending,
            result: None,
            max_loops: config.max_loops,
        }
    }

    // Set LLM
    pub fn set_llm(&mut self, llm: Arc<dyn LLM>) {
        self.context.set_llm(llm);
    }

    // Inject system prompt
    pub fn inject_system_prompt(&mut self, prompt: String) {
        self.context.inject_system_prompt(prompt);
    }

    // Execute SubAgent
    pub async fn run(
        &mut self,
        llm: Arc<dyn LLM>,
        tool_registry: Arc<ToolRegistry>,
        mut ctrlc_rx: Option<watch::Receiver<u64>>,
    ) -> Result<String> {
        info!(
            "SubAgent {} start: type={}, max_loops={}",
            self.id, self.agent_type, self.max_loops
        );
        self.context.set_llm(llm.clone());
        // Add task message into context
        self.context.add_message(crate::llm::Message {
            role: crate::llm::Role::User,
            content: self.task.clone(),
        });

        self.status = SubAgentStatus::Running;
        let mut loop_count = 0;

        loop {
            loop_count += 1;
            debug!(
                "SubAgent {} loop {}/{}",
                self.id, loop_count, self.max_loops
            );

            let response = match ctrlc_rx.as_mut() {
                Some(rx) => {
                    tokio::select! {
                        res = llm.complete(self.context.get_history()) => res?,
                        _ = rx.changed() => {
                            self.status = SubAgentStatus::Failed("Cancelled by user".to_string());
                            warn!("SubAgent {} cancelled by user", self.id);
                            return Err(anyhow::anyhow!("SubAgent {} cancelled", self.id));
                        }
                    }
                }
                None => llm.complete(self.context.get_history()).await?,
            };
            self.context.add_message(crate::llm::Message {
                role: crate::llm::Role::Assistant,
                content: response.clone(),
            });

            // Check for tool calls
            if let Some(tool_call) = parse_tool_call(&response) {
                info!("SubAgent {} tool call: {}", self.id, tool_call.name);
                let output = match tool_registry.get(&tool_call.name) {
                    Some(tool) => {
                        let call = tool.call(tool_call.args);
                        match ctrlc_rx.as_mut() {
                            Some(rx) => {
                                tokio::select! {
                                    res = call => match res {
                                        Ok(o) => o,
                                        Err(e) => {
                                            warn!(
                                                "SubAgent {} tool error: {} -> {}",
                                                self.id, tool_call.name, e
                                            );
                                            format!("Error: {}", e)
                                        }
                                    },
                                    _ = rx.changed() => {
                                        self.status = SubAgentStatus::Failed("Cancelled by user".to_string());
                                        warn!("SubAgent {} cancelled by user", self.id);
                                        return Err(anyhow::anyhow!("SubAgent {} cancelled", self.id));
                                    }
                                }
                            }
                            None => match call.await {
                                Ok(o) => o,
                                Err(e) => {
                                    warn!(
                                        "SubAgent {} tool error: {} -> {}",
                                        self.id, tool_call.name, e
                                    );
                                    format!("Error: {}", e)
                                }
                            },
                        }
                    }
                    None => format!("Tool '{}' not found", tool_call.name),
                };
                debug!(
                    "SubAgent {} tool {} output: {}",
                    self.id, tool_call.name, output
                );

                // Add tool output into context
                self.context.add_message(crate::llm::Message {
                    role: crate::llm::Role::User,
                    content: format!("Tool '{}' output:\n{}", tool_call.name, output),
                });
            } else {
                // No tool call. If model indicates completion, stop. Otherwise continue.
                if let Some(final_text) = extract_final(&response) {
                    self.status = SubAgentStatus::Completed;
                    self.result = Some(final_text.clone());
                    info!("SubAgent {} completed", self.id);
                    return Ok(final_text);
                }

                if loop_count >= self.max_loops {
                    self.status = SubAgentStatus::Failed("Max loops reached".to_string());
                    warn!("SubAgent {} failed: max loops reached", self.id);
                    return Err(anyhow::anyhow!("SubAgent {} max loops reached", self.id));
                }

                self.context.add_message(crate::llm::Message {
                    role: crate::llm::Role::User,
                    content: "Continue. If finished, wrap the final answer in <final>...</final>."
                        .to_string(),
                });
            }

            // Try to compress context to avoid unbounded growth
            if let Err(e) = self.context.compress().await {
                debug!("SubAgent {} context compress failed: {}", self.id, e);
            }
        }
    }
}

// SubAgent manager
#[derive(Clone)]
pub struct SubAgentManager {
    agents: Arc<Mutex<HashMap<String, Arc<AsyncMutex<SubAgent>>>>>,
    llm: Arc<dyn LLM>,
    shared_tool_registry: Arc<ToolRegistry>,
}

impl SubAgentManager {
    pub fn new(llm: Arc<dyn LLM>, tool_registry: Arc<ToolRegistry>) -> Self {
        Self {
            agents: Arc::new(Mutex::new(HashMap::new())),
            llm,
            shared_tool_registry: tool_registry,
        }
    }

    // Create a new SubAgent and return its ID
    pub fn spawn(&mut self, config: SubAgentConfig) -> Result<String> {
        let mut agent = SubAgent::new(config);
        agent.set_llm(self.llm.clone());

        // Inject system prompt by agent type
        let mut prompt = Self::generate_system_prompt(&agent.agent_type);
        prompt.push_str("\n\n");
        prompt.push_str("When you are finished, wrap the final answer in <final>...</final>.\n");
        prompt.push_str("If you need more steps and no tool call is required, continue until you are ready to finalize.\n\n");
        prompt.push_str(&self.shared_tool_registry.generate_tool_instructions());
        agent.inject_system_prompt(prompt);

        let id = agent.id.clone();
        let mut agents = self.agents.lock().expect("SubAgentManager lock poisoned");
        agents.insert(id.clone(), Arc::new(AsyncMutex::new(agent)));
        Ok(id)
    }

    // Get SubAgent by ID
    pub fn get(&self, id: &str) -> Option<Arc<AsyncMutex<SubAgent>>> {
        let agents = self.agents.lock().expect("SubAgentManager lock poisoned");
        agents.get(id).cloned()
    }

    // Mark a SubAgent as cancelled
    pub async fn cancel(&self, id: &str, reason: &str) {
        if let Some(agent) = self.get(id) {
            let mut locked = agent.lock().await;
            if matches!(
                locked.status,
                SubAgentStatus::Pending | SubAgentStatus::Running
            ) {
                locked.status = SubAgentStatus::Failed(reason.to_string());
                locked.result = None;
            }
        }
    }

    // Generate system prompt for a given type
    fn generate_system_prompt(agent_type: &str) -> String {
        match agent_type.to_lowercase().as_str() {
            "code" => Self::code_agent_prompt(),
            "test" => Self::test_agent_prompt(),
            "doc" => Self::doc_agent_prompt(),
            "analysis" => Self::analysis_agent_prompt(),
            _ => Self::dynamic_agent_prompt(),
        }
    }

    fn code_agent_prompt() -> String {
        String::from(
            "You are a Code SubAgent focused on code implementation, refactoring, and optimization.\n\
            Guidelines:\n\
            - Write clean, idiomatic code\n\
            - Follow existing code patterns and conventions\n\
            - Add comments for complex logic\n\
            - Consider edge cases and error handling\n\
            - Run tests to verify your changes\n\n\
            You have access to bash tool for running commands and testing.",
        )
    }

    fn test_agent_prompt() -> String {
        String::from(
            "You are a Test SubAgent focused on writing and improving tests.\n\
            Guidelines:\n\
            - Write comprehensive unit tests\n\
            - Cover edge cases and error scenarios\n\
            - Use appropriate testing frameworks\n\
            - Ensure tests are fast and isolated\n\
            - Provide clear test documentation\n\n\
            You have access to bash tool for running tests.",
        )
    }

    fn doc_agent_prompt() -> String {
        String::from(
            "You are a Documentation SubAgent focused on creating and improving documentation.\n\
            Guidelines:\n\
            - Write clear, concise documentation\n\
            - Include code examples where appropriate\n\
            - Document public APIs thoroughly\n\
            - Keep documentation up-to-date with code changes\n\
            - Use markdown format for readability\n\n\
            You have access to bash tool for reading files and checking documentation.",
        )
    }

    fn analysis_agent_prompt() -> String {
        String::from(
            "You are an Analysis SubAgent focused on understanding and analyzing codebases.\n\
            Guidelines:\n\
            - Analyze code structure and architecture\n\
            - Identify patterns and anti-patterns\n\
            - Provide insights on code quality\n\
            - Suggest improvements where needed\n\
            - Be thorough in your analysis\n\n\
            You have access to bash tool for exploring the codebase.",
        )
    }

    fn dynamic_agent_prompt() -> String {
        String::from(
            "You are a general-purpose SubAgent.\n\
            Guidelines:\n\
            - Focus on completing the assigned task\n\
            - Ask for clarification if needed\n\
            - Provide clear, actionable results\n\
            - Report any errors or blockers encountered\n\n\
            You have access to bash tool for executing commands.",
        )
    }
}

// Parse tool call
pub fn parse_tool_call(content: &str) -> Option<ToolCall> {
    let re = regex::Regex::new(r"(?s)<tool_code>\s*(.*?)\s*</tool_code>").ok()?;
    if let Some(caps) = re.captures(content) {
        let json_str = caps.get(1)?.as_str();
        #[derive(serde::Deserialize)]
        struct RawToolCall {
            name: String,
            args: Value,
        }
        if let Ok(raw) = serde_json::from_str::<RawToolCall>(json_str) {
            return Some(ToolCall {
                name: raw.name,
                args: raw.args,
            });
        }
    }
    None
}

fn extract_final(content: &str) -> Option<String> {
    let re = regex::Regex::new(r"(?s)<final>\s*(.*?)\s*</final>").ok()?;
    let caps = re.captures(content)?;
    let text = caps.get(1)?.as_str().trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

// Parse parallel task configs
pub fn parse_parallel_tasks(content: &str) -> Option<Vec<SubAgentConfig>> {
    let re = regex::Regex::new(r"(?s)<parallel>\s*(.*?)\s*</parallel>").ok()?;
    let inner = re.captures(content)?.get(1)?.as_str();

    let re_task = regex::Regex::new(r"\{[^{}]*\}").ok()?;
    let mut configs = Vec::new();

    for cap in re_task.find_iter(inner) {
        let json_str = cap.as_str();
        if let Ok(value) = serde_json::from_str::<Value>(json_str)
            && let Some(task) = value["task"].as_str()
        {
            let agent_type = value["type"].as_str().unwrap_or("dynamic").to_string();
            let max_loops = value["max_loops"].as_u64().unwrap_or(20) as usize;

            configs.push(SubAgentConfig::new(task.to_string(), agent_type, max_loops));
        }
    }

    if configs.is_empty() {
        None
    } else {
        Some(configs)
    }
}
