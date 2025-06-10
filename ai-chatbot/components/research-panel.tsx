'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { CrossIcon } from './icons';
import { Button } from './ui/button';
import { Markdown } from './markdown';

interface ResearchPanelProps {
  isOpen: boolean;
  onClose: () => void;
  searchResults: Array<{
    content: string;
    metadata: {
      hybrid_score: number;
      entities: string[];
      topics: string[];
      search_methods: {
        vector: boolean;
        sparse: boolean;
        graph: boolean;
      };
    };
  }>;
}

export function ResearchPanel({ isOpen, onClose, searchResults }: ResearchPanelProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed right-0 top-0 h-dvh w-[400px] bg-background border-l border-border z-50"
          initial={{ x: 400 }}
          animate={{ x: 0 }}
          exit={{ x: 400 }}
          transition={{ type: 'spring', damping: 30, stiffness: 200 }}
        >
          <div className="flex flex-col h-full">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="text-lg font-semibold">Research Results</h2>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <CrossIcon />
              </Button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {searchResults.map((result, index) => (
                <div key={index} className="p-4 rounded-lg border bg-muted/50">
                  <div className="mb-2">
                    <Markdown>{result.content}</Markdown>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    <div>Score: {result.metadata.hybrid_score.toFixed(2)}</div>
                    <div>Search Methods: {
                      Object.entries(result.metadata.search_methods)
                        .filter(([_, enabled]) => enabled)
                        .map(([method]) => method)
                        .join(', ')
                    }</div>
                    {result.metadata.entities.length > 0 && (
                      <div>Entities: {result.metadata.entities.join(', ')}</div>
                    )}
                    {result.metadata.topics.length > 0 && (
                      <div>Topics: {result.metadata.topics.join(', ')}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
} 