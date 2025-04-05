"use client";
import { CodeBlock } from "@/components/ui/code-block";
import ReactMarkdown from "react-markdown";
import CopyButton from "../ui/copy-button";

// Example CodeBlock component (you can enhance it with syntax highlighting)
const CodeBlockComponent = ({ language, code }: any) => (
  <div className="my-2">
    <CodeBlock
      language={language}
      filename=" "
      // highlightLines={[9, 13, 14, 18]}
      code={code}
    />
  </div>
);

// Utility function to parse the API response into segments
// This function splits the response on triple backticks
const parseResponse = (response: any) => {
  const segments = [];
  const regex = /```(\w+)?\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(response)) !== null) {
    // Text before the code block
    if (match.index > lastIndex) {
      segments.push({
        type: "text",
        content: response.slice(lastIndex, match.index),
      });
    }
    // The code block segment
    segments.push({
      type: "code",
      language: match[1] || "plaintext",
      content: match[2],
    });
    lastIndex = regex.lastIndex;
  }
  // Remainder text after last code block
  if (lastIndex < response.length) {
    segments.push({
      type: "text",
      content: response.slice(lastIndex),
    });
  }
  return segments;
};

const LLMResponse = ({ title, content }: any) => {
  // Parse the response into segments
  const segments = parseResponse(content);

  // Copy the entire container's HTML (including formatting) to clipboard
  const copyContentToClipboard = async () => {
    const contentDiv = document.getElementById(`${title}-content`);
    if (contentDiv) {
      const htmlContent = contentDiv.innerHTML;
      try {
        // Create a Blob from the HTML content
        const blob = new Blob([htmlContent], { type: "text/html" });
        // Write the HTML blob to the clipboard using ClipboardItem
        await navigator.clipboard.write([
          new ClipboardItem({
            "text/html": blob,
          }),
        ]);
        alert("Content copied to clipboard with formatting!");
      } catch (error) {
        console.error("Copy failed:", error);
        alert("Failed to copy content.");
      }
    }
  };

  return (
    <div className="w-2xl mx-auto">
      <div
        id={`${title}-content`}
        className="w-2xl rounded-3xl bg-neutral-200 dark:bg-neutral-900 py-7 px-8 m-auto"
      >
        <p className="text-3xl mb-1">{title}</p>
        {segments.map((segment, index) =>
          segment.type === "text" ? (
            <ReactMarkdown key={index}>{segment.content}</ReactMarkdown>
          ) : (
            <CodeBlockComponent
              key={index}
              language={segment.language}
              code={segment.content}
            />
          )
        )}
      </div>
      <CopyButton copyToClipboard={copyContentToClipboard} />
    </div>
  );
};

export default LLMResponse;
