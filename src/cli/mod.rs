use crate::context::ContextManager;
use crate::llm::{LLM, Message, Role};
use crate::session::SessionManager;
use crate::subagent::{SubAgentConfig, SubAgentManager, parse_parallel_tasks};
use crate::tool::ToolRegistry;
use anyhow::Result;
use colored::Colorize;
use log::{debug, error, info};
use regex::Regex;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::watch;

const DEFAULT_CMD_HISTORY_FILE_NAME: &str = "__history";

pub struct Cli {
    context: ContextManager,
    session_manager: SessionManager,
    tool_registry: Arc<ToolRegistry>,
    llm: Arc<dyn LLM>,
    max_loops: usize,
    subagent_manager: SubAgentManager,
}

impl Cli {
    pub fn new(
        context: ContextManager,
        session_manager: SessionManager,
        tool_registry: ToolRegistry,
        llm: Arc<dyn LLM>,
        max_loops: usize,
    ) -> Self {
        let tool_registry = Arc::new(tool_registry);
        let subagent_manager = SubAgentManager::new(llm.clone(), tool_registry.clone());

        Self {
            context,
            session_manager,
            tool_registry,
            llm,
            max_loops,
            subagent_manager,
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        println!("Mini Agent CLI-Type /help for commands");

        // inject System Prompt
        let system_prompt = self.tool_registry.generate_system_prompt();
        self.context.inject_system_prompt(system_prompt);

        let (ctrlc_tx, mut ctrlc_rx) = watch::channel(0u64);
        let ctrlc_tx_signal = ctrlc_tx.clone();
        tokio::spawn(async move {
            let mut n = 0u64;
            loop {
                if tokio::signal::ctrl_c().await.is_ok() {
                    n = n.saturating_add(1);
                    let _ = ctrlc_tx_signal.send(n);
                }
            }
        });
        let mut rl = DefaultEditor::new()?;
        // Optionally load history if file exists
        if rl.load_history(DEFAULT_CMD_HISTORY_FILE_NAME).is_err() {
            // No history file yet
        }

        loop {
            let readline = rl.readline(">> ");
            match readline {
                Ok(line) => {
                    let input = line.trim();
                    if input.is_empty() {
                        continue;
                    }
                    rl.add_history_entry(line.as_str())?;
                    if input.starts_with('/') {
                        match input.split_whitespace().next() {
                            Some("/quit") | Some("/exit") => break,
                            Some("/help") => {
                                println!("Commands:");
                                println!("  /save <name> - Save session");
                                println!("  /load <name> - Load session");
                                println!("  /list - List sessions");
                                println!("  /clear - Clear context");
                                println!("  /tools - List tools");
                                println!("  /quit - Exit");
                            }
                            Some("/save") => {
                                let parts: Vec<&str> = input.split_whitespace().collect();
                                if parts.len() > 1 {
                                    if let Err(e) =
                                        self.session_manager.save_session(parts[1], &self.context)
                                    {
                                        println!("Error saving session:{}", e);
                                    } else {
                                        println!("Session saved as:{}", parts[1]);
                                    }
                                } else {
                                    println!("Usage: /save <name>");
                                }
                            }
                            Some("/load") => {
                                let parts: Vec<&str> = input.split_whitespace().collect();
                                if parts.len() > 1 {
                                    if let Err(e) = self
                                        .session_manager
                                        .load_session(parts[1], &mut self.context)
                                    {
                                        println!("Error loading session: {}", e);
                                    } else {
                                        println!("Session loaded");
                                        // Re-inject system prompt after loading to ensure tools are current
                                        let system_prompt =
                                            self.tool_registry.generate_system_prompt();
                                        self.context.inject_system_prompt(system_prompt);
                                    }
                                } else {
                                    println!("Usage: /load <name>");
                                }
                            }
                            Some("/list") => match self.session_manager.list_sessions() {
                                Ok(sessions) => {
                                    println!("Sessions: {:?}", sessions);
                                }
                                Err(e) => println!("Error listing sessions: {}", e),
                            },
                            Some("/clear") => {
                                self.context.clear_history();
                                // Re-inject system prompt
                                let system_prompt = self.tool_registry.generate_system_prompt();
                                self.context.inject_system_prompt(system_prompt);
                                println!("Context cleared");
                            }
                            Some("/tools") => {
                                for tool in self.tool_registry.list() {
                                    println!("- {}:{}", tool.name(), tool.description());
                                }
                            }
                            _ => println!("Unknown command.Type /help"),
                        }
                        continue;
                    }

                    // Chat with LLM
                    let _ = ctrlc_rx.borrow_and_update();
                    self.context.add_message(Message {
                        role: Role::User,
                        content: input.to_string(),
                    });

                    let mut agent_loop_count = 0;

                    while agent_loop_count < self.max_loops {
                        agent_loop_count += 1;

                        let mut ctrl_c = ctrlc_rx.clone();
                        let response = tokio::select! {
                            res = self.llm.complete(self.context.get_history()) => res,
                            _ = ctrl_c.changed() => {
                                println!("CTRL-C");
                                break;
                            }
                        };

                        let response = match response {
                            Ok(r) => r,
                            Err(e) => {
                                println!("Error: {}", e);
                                break;
                            }
                        };

                        print_formatted_response(&response);
                        self.context.add_message(Message {
                            role: Role::Assistant,
                            content: response.clone(),
                        });

                        if let Some(tool_call) = parse_tool_call(&response) {
                            println!(">> Executing tool: {}...", &tool_call.name);
                            debug!("Tool call parsed: {:?}", tool_call);

                            // Check if this is a subagent tool call
                            let is_subagent_tool = matches!(
                                tool_call.name.as_str(),
                                "subagent" | "code_subagent" | "test_subagent" | "doc_subagent"
                            );

                            let output = if is_subagent_tool {
                                self.handle_subagent_call(&tool_call, &ctrlc_rx).await
                            } else {
                                let mut interrupted = false;
                                let mut ctrl_c = ctrlc_rx.clone();
                                let output = tokio::select! {
                                    res = async {
                                        match self.tool_registry.get(&tool_call.name) {
                                            Some(tool) => match tool.call(tool_call.args).await {
                                                Ok(o) => o,
                                                Err(e) => format!("Error executing tool: {}", e),
                                            },
                                            None => format!("Error: Tool '{}' not found", tool_call.name),
                                        }
                                    } => res,
                                    _ = ctrl_c.changed() => {
                                        interrupted = true;
                                        String::new()
                                    }
                                };

                                if interrupted {
                                    println!("CTRL-C");
                                    break;
                                }

                                output
                            };

                            println!(">> Tool Output:\n{}", output.trim());
                            self.context.add_message(Message {
                                role: Role::User, // Treating tool output as User message for simplicity/compatibility
                                content: format!("Tool '{}' output:\n{}", tool_call.name, output),
                            });

                            // Check for parallel tasks
                            if let Some(configs) = parse_parallel_tasks(&response) {
                                info!("Found {} parallel tasks", configs.len());
                                let results = self.handle_parallel_tasks(configs, &ctrlc_rx).await;
                                self.context.add_message(Message {
                                    role: Role::User,
                                    content: format!(
                                        "Parallel tasks results:\n{}",
                                        results.join("\n---\n")
                                    ),
                                });
                            }

                            // Continue loop
                        } else if let Some(final_text) = extract_final(&response) {
                            self.context.add_message(Message {
                                role: Role::Assistant,
                                content: final_text,
                            });
                            break;
                        } else {
                            self.context.add_message(Message {
                                role: Role::User,
                                content: "Continue. If finished, wrap the final answer in <final>...</final>."
                                    .to_string(),
                            });
                        }
                    }

                    // Try to compress context
                    if let Err(e) = self.context.compress().await {
                        println!("(Context compression error: {})", e);
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("CTRL-C");
                    let _ = ctrlc_tx.send(ctrlc_tx.borrow().saturating_add(1));
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("CTRL-D");
                    break;
                }
                Err(err) => {
                    println!("Error: {:?}", err);
                    break;
                }
            }
        }

        // Save history on exit
        rl.save_history(DEFAULT_CMD_HISTORY_FILE_NAME)?;
        Ok(())
    }

    // Handle subagent tool calls
    async fn handle_subagent_call(
        &mut self,
        tool_call: &ToolCall,
        ctrlc_rx: &watch::Receiver<u64>,
    ) -> String {
        let task = match tool_call.args.get("task") {
            Some(v) => v.as_str().unwrap_or(""),
            None => return "Error: Missing 'task' argument for subagent".to_string(),
        };

        let agent_type = tool_call
            .args
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("dynamic");

        let max_loops = tool_call
            .args
            .get("max_loops")
            .and_then(|v| v.as_u64())
            .unwrap_or(20) as usize;

        info!("Executing subagent: type={}, task={}", agent_type, task);

        let config = SubAgentConfig::new(task.to_string(), agent_type.to_string(), max_loops);

        let id = match self.subagent_manager.spawn(config) {
            Ok(id) => id,
            Err(e) => return format!("SubAgent [{}] failed to spawn: {}", agent_type, e),
        };

        let agent = match self.subagent_manager.get(&id) {
            Some(agent) => agent,
            None => return format!("SubAgent [{}] not found after spawn", agent_type),
        };

        let llm = self.llm.clone();
        let tool_registry = self.tool_registry.clone();
        let ctrlc_for_task = ctrlc_rx.clone();
        let mut handle = tokio::spawn(async move {
            let mut locked = agent.lock().await;
            locked.run(llm, tool_registry, Some(ctrlc_for_task)).await
        });

        let mut ctrl_c = ctrlc_rx.clone();
        tokio::select! {
            res = &mut handle => {
                match res {
                    Ok(Ok(result)) => format!("SubAgent [{}] completed:\n{}", agent_type, result),
                    Ok(Err(e)) => format!("SubAgent [{}] failed: {}", agent_type, e),
                    Err(e) => format!("SubAgent [{}] join error: {}", agent_type, e),
                }
            }
            _ = ctrl_c.changed() => {
                handle.abort();
                self.subagent_manager.cancel(&id, "Cancelled by user").await;
                format!("SubAgent [{}] cancelled by user", agent_type)
            }
        }
    }

    // Handle parallel task execution - truly parallel
    async fn handle_parallel_tasks(
        &mut self,
        configs: Vec<SubAgentConfig>,
        ctrlc_rx: &watch::Receiver<u64>,
    ) -> Vec<String> {
        info!("Executing {} parallel tasks", configs.len());

        if configs.is_empty() {
            return vec![];
        }

        // Spawn all subagents in parallel
        let mut results = Vec::new();
        let mut join_set = tokio::task::JoinSet::new();
        let mut pending: HashMap<String, (String, String)> = HashMap::new();

        for config in configs {
            let agent_type = config.agent_type.clone();
            let task = config.task.clone();

            let id = match self.subagent_manager.spawn(config) {
                Ok(id) => id,
                Err(e) => {
                    results.push(format!("[{}] ERROR: {} - {}", agent_type, task, e));
                    continue;
                }
            };

            let agent = match self.subagent_manager.get(&id) {
                Some(agent) => agent,
                None => {
                    results.push(format!("[{}] ERROR: {} - missing agent", agent_type, task));
                    continue;
                }
            };

            pending.insert(id.clone(), (agent_type.clone(), task.clone()));
            let llm = self.llm.clone();
            let tool_registry = self.tool_registry.clone();
            let ctrlc_for_task = ctrlc_rx.clone();

            join_set.spawn(async move {
                let mut locked = agent.lock().await;
                let res = locked.run(llm, tool_registry, Some(ctrlc_for_task)).await;
                (id, agent_type, task, res)
            });
        }

        let mut ctrl_c = ctrlc_rx.clone();
        loop {
            tokio::select! {
                _ = ctrl_c.changed() => {
                    join_set.abort_all();
                    for (id, (agent_type, task)) in pending.drain() {
                        self.subagent_manager.cancel(&id, "Cancelled by user").await;
                        results.push(format!("[{}] CANCELLED: {}", agent_type, task));
                    }
                    break;
                }
                res = join_set.join_next() => {
                    match res {
                        Some(Ok((id, agent_type, task, Ok(result)))) => {
                            pending.remove(&id);
                            results.push(format!("[{}] Task: {}\nResult: {}", agent_type, task, result));
                        }
                        Some(Ok((id, agent_type, task, Err(e)))) => {
                            pending.remove(&id);
                            results.push(format!("[{}] ERROR: {} - {}", agent_type, task, e));
                        }
                        Some(Err(e)) => {
                            results.push(format!("JOIN ERROR: {}", e));
                        }
                        None => break,
                    }
                }
            }
        }

        results
    }
}

fn print_formatted_response(response: &str) {
    let re = Regex::new(r"(?s)(<thinking>.*?</thinking>)|(<tool_code>.*?</tool_code>)").unwrap();

    let mut last_end = 0;

    for cap in re.find_iter(response) {
        // Print text before match
        if cap.start() > last_end {
            println!("{}", &response[last_end..cap.start()]);
        }

        let matched_text = cap.as_str();
        if matched_text.starts_with("<thinking>") {
            // Print thinking in dim/gray
            println!("{}", matched_text.blue());
        } else if matched_text.starts_with("<tool_code>") {
            // Print tool code in yellow
            println!("{}\n", matched_text.yellow());
        }

        last_end = cap.end();
    }

    // Print remaining text
    if last_end < response.len() {
        println!("{}", &response[last_end..]);
    }

    println!(); // Newline
}

fn extract_final(content: &str) -> Option<String> {
    let re = Regex::new(r"(?s)<final>\s*(.*?)\s*</final>").ok()?;
    let caps = re.captures(content)?;
    let text = caps.get(1)?.as_str().trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

#[derive(Debug)]
struct ToolCall {
    name: String,
    args: Value,
}

fn parse_tool_call(content: &str) -> Option<ToolCall> {
    let re = regex::Regex::new(r"(?s)<tool_code>\s*(.*?)\s*</tool_code>").ok()?;
    if let Some(caps) = re.captures(content) {
        let json_str = caps.get(1)?.as_str();
        #[derive(serde::Deserialize)]
        struct RawToolCall {
            name: String,
            args: Value,
        }
        return match serde_json::from_str::<RawToolCall>(json_str) {
            Ok(raw) => Some(ToolCall {
                name: raw.name,
                args: raw.args,
            }),
            Err(e) => {
                error!("Failed to deserialize tool call JSON: {}, {e:?}", json_str);
                None
            }
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_call() {
        let content = "I will use the bash tool.\n<tool_code>\n{\n  \"name\": \"bash\",\n  \"args\": {\n    \"command\": \"ls\"\n  }\n}\n</tool_code>";
        let tool_call = parse_tool_call(content).unwrap();
        assert_eq!(tool_call.name, "bash");
        assert_eq!(tool_call.args["command"], "ls");
    }

    #[test]
    fn test_parse_tool_call_multiline() {
        let content = r#"<tool_code>
{
  "name": "bash",
  "args": {
    "command": "ls -la"
  }
}
</tool_code>"#;
        let tool_call = parse_tool_call(content).unwrap();
        assert_eq!(tool_call.name, "bash");
    }

    #[test]
    fn test_parse_no_tool_call() {
        let content = "Just a normal message";
        assert!(parse_tool_call(content).is_none());
    }
}
