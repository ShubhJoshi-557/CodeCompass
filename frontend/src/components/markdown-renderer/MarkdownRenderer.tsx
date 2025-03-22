"use client";
import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import CopyButton from "../ui/copy-button";
import { CodeBlockDemo } from "../code-block/CodeBlock";

const MarkdownRenderer = ({ title, content }: any) => {
  return (
    <div className="markdown-body w-3xl rounded-xl bg-neutral-200 dark:bg-neutral-900 p-8 m-auto">
      <div className="flex justify-between">
        <p className="mb-5 text-3xl font-semibold">{title}</p>
        <CopyButton className="ml-auto" content={content} />
      </div>

      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize]}
      >
        {content}
      </ReactMarkdown>
      <CodeBlockDemo />
    </div>
  );
};

export default MarkdownRenderer;
