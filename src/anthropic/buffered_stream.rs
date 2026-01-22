//! 带缓冲的流处理上下文
//!
//! 包装上游 StreamContext，控制事件缓冲和发送时机。
//! 在收到 ContextUsage 事件之前缓冲所有事件，收到后一次性发送。

use crate::kiro::model::events::Event;

use super::stream::{SseEvent, StreamContext};

/// 上下文窗口大小（200k tokens）
const CONTEXT_WINDOW_SIZE: i32 = 200_000;

/// 缓冲状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// 等待 contextUsageEvent
    WaitingForContext,
    /// 正常发送
    Streaming,
}

/// 带缓冲的流处理上下文
pub struct BufferedStreamContext {
    /// 上游 StreamContext（不改动）
    pub inner: StreamContext,
    /// 存起来的事件
    pub buffer: Vec<SseEvent>,
    /// 当前缓冲状态
    state: BufferState,
}

impl BufferedStreamContext {
    /// 创建新的缓冲流上下文
    pub fn new(model: impl Into<String>, input_tokens: i32, thinking_enabled: bool) -> Self {
        Self {
            inner: StreamContext::new_with_thinking(model, input_tokens, thinking_enabled),
            buffer: Vec::new(),
            state: BufferState::WaitingForContext,
        }
    }

    /// 处理 Kiro 事件
    ///
    /// 在 WaitingForContext 状态下，所有事件都会被缓冲。
    /// 收到 ContextUsage 事件后，触发 flush 并切换到 Streaming 状态。
    pub fn process_event(&mut self, event: &Event) -> Vec<SseEvent> {
        match event {
            Event::ContextUsage(context_usage) => {
                let actual_input_tokens = (context_usage.context_usage_percentage
                    * (CONTEXT_WINDOW_SIZE as f64)
                    / 100.0) as i32;
                
                tracing::debug!(
                    "收到 contextUsageEvent: {}%, 计算 input_tokens: {}",
                    context_usage.context_usage_percentage,
                    actual_input_tokens
                );

                self.inner.context_input_tokens = Some(actual_input_tokens);
                self.inner.input_tokens = actual_input_tokens;

                self.flush_pending()
            }
            _ => {
                let events = self.inner.process_kiro_event(event);
                
                match self.state {
                    BufferState::WaitingForContext => {
                        self.buffer.extend(events);
                        Vec::new()
                    }
                    BufferState::Streaming => events,
                }
            }
        }
    }

    /// Flush 缓冲的事件
    ///
    /// 生成 message_start 事件（使用真实的 input_tokens），
    /// 然后发送所有缓冲的事件，最后切换到 Streaming 状态。
    pub fn flush_pending(&mut self) -> Vec<SseEvent> {
        if self.state == BufferState::Streaming {
            return Vec::new();
        }

        let mut events = Vec::new();

        let msg_start = self.inner.create_message_start_event();
        if let Some(event) = self.inner.state_manager.handle_message_start(msg_start) {
            events.push(event);
        }

        events.append(&mut self.buffer);

        self.state = BufferState::Streaming;

        events
    }

    /// 生成最终事件序列
    pub fn generate_final_events(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();

        if self.state == BufferState::WaitingForContext {
            events.extend(self.flush_pending());
        }

        events.extend(self.inner.generate_final_events());

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiro::model::events::{AssistantResponseEvent, ContextUsageEvent};

    #[test]
    fn test_buffer_events_until_context_usage() {
        let mut ctx = BufferedStreamContext::new("test-model", 1000, false);

        let text_event = Event::AssistantResponse(AssistantResponseEvent {
            content: "Hello".to_string(),
        });
        let events = ctx.process_event(&text_event);
        assert_eq!(events.len(), 0, "events should be buffered");
        assert!(!ctx.buffer.is_empty(), "buffer should contain events");
    }

    #[test]
    fn test_flush_on_context_usage() {
        let mut ctx = BufferedStreamContext::new("test-model", 1000, false);

        let text_event = Event::AssistantResponse(AssistantResponseEvent {
            content: "Hello".to_string(),
        });
        ctx.process_event(&text_event);

        let context_event = Event::ContextUsage(ContextUsageEvent {
            context_usage_percentage: 50.0,
        });
        let events = ctx.process_event(&context_event);

        assert!(!events.is_empty(), "should flush events");
        assert!(
            events.iter().any(|e| e.event == "message_start"),
            "should include message_start"
        );
        assert_eq!(ctx.inner.input_tokens, 100_000, "should use real input_tokens");
    }

    #[test]
    fn test_streaming_after_flush() {
        let mut ctx = BufferedStreamContext::new("test-model", 1000, false);

        let context_event = Event::ContextUsage(ContextUsageEvent {
            context_usage_percentage: 25.0,
        });
        ctx.process_event(&context_event);

        let text_event = Event::AssistantResponse(AssistantResponseEvent {
            content: "World".to_string(),
        });
        let events = ctx.process_event(&text_event);

        assert!(!events.is_empty(), "should stream events directly");
    }

    #[test]
    fn test_input_tokens_calculation() {
        let mut ctx = BufferedStreamContext::new("test-model", 1000, false);

        let context_event = Event::ContextUsage(ContextUsageEvent {
            context_usage_percentage: 10.5,
        });
        ctx.process_event(&context_event);

        let expected = (10.5 * 200_000.0 / 100.0) as i32;
        assert_eq!(ctx.inner.input_tokens, expected);
        assert_eq!(ctx.inner.context_input_tokens, Some(expected));
    }
}
