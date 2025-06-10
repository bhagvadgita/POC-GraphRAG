'use client';

import { useState } from 'react';
import { ChatHeader } from '@/components/chat-header';
import { MultimodalInput } from './multimodal-input';
import { Messages } from './messages';
import type { VisibilityType } from './visibility-selector';
import { useChatVisibility } from '@/hooks/use-chat-visibility';
import { generateUUID } from '@/lib/utils';
import type { Session } from 'next-auth';

export function Chat({
  id,
  initialMessages,
  initialChatModel,
  initialVisibilityType,
  isReadonly,
  session,
}: {
  id: string;
  initialMessages: Array<any>;
  initialChatModel: string;
  initialVisibilityType: VisibilityType;
  isReadonly: boolean;
  session: Session;
}) {
  const { visibilityType } = useChatVisibility({
    chatId: id,
    initialVisibilityType,
  });

  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState<'ready' | 'submitted' | 'streaming' | 'error'>('ready');
  const [selectedModel, setSelectedModel] = useState(initialChatModel);
  const [searchResults, setSearchResults] = useState<Record<string, any[]>>({});

  const handleModelChange = (modelId: string) => {
    if (modelId === selectedModel) return;
    setSelectedModel(modelId);
  };

  const handleSubmit = async (event?: { preventDefault?: () => void }) => {
    event?.preventDefault?.();
    if (!input.trim()) return;

    // Add user message
    const userMessage = {
      id: generateUUID(),
      role: 'user',
      content: input,
      parts: [{ type: 'text', text: input }],
      createdAt: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setStatus('submitted');

    try {
      // Send to chat-query API
      const response = await fetch('http://localhost:8000/chat-query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: input,
          k: 3,
          search_type: selectedModel === 'hybrid-search' ? 'hybrid' : 
                      selectedModel === 'graph-search' ? 'graph' : 
                      selectedModel === 'vector-search' ? 'vector' :
                      selectedModel === 'sparse-search' ? 'sparse' : 'hybrid'
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Add assistant message
      const assistantMessage = {
        id: generateUUID(),
        role: 'assistant',
        content: data.model_response || '',
        parts: [{ type: 'text', text: data.model_response || '' }],
        createdAt: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
      
      // Store search results for this message
      setSearchResults(prev => ({
        ...prev,
        [assistantMessage.id]: data.search_results
      }));
    } catch (error) {
      console.error('Error:', error);
      // Add error message
      const errorMessage = {
        id: generateUUID(),
        role: 'assistant',
        content: 'Sorry, there was an error processing your request.',
        parts: [{ type: 'text', text: 'Sorry, there was an error processing your request.' }],
        createdAt: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      setStatus('error');
    } finally {
      setStatus('ready');
    }
  };

  return (
    <>
      <div className="flex flex-col min-w-0 h-dvh bg-background">
        <ChatHeader
          chatId={id}
          selectedModelId={selectedModel}
          selectedVisibilityType={initialVisibilityType}
          isReadonly={isReadonly}
          session={session}
          onModelChange={handleModelChange}
        />

        <Messages
          chatId={id}
          status={status}
          messages={messages}
          setMessages={setMessages}
          reload={async () => null}
          isReadonly={isReadonly}
          isArtifactVisible={false}
          votes={[]}
          searchResults={searchResults}
        />

        <form className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl" onSubmit={handleSubmit}>
          {!isReadonly && (
            <MultimodalInput
              chatId={id}
              input={input}
              setInput={setInput}
              handleSubmit={handleSubmit}
              status={status}
              stop={() => {}}
              attachments={[]}
              setAttachments={() => {}}
              messages={messages}
              setMessages={setMessages}
              append={async (message) => {
                setMessages(prev => [...prev, message]);
                return Promise.resolve(null);
              }}
              selectedVisibilityType={visibilityType}
            />
          )}
        </form>
      </div>
    </>
  );
}
