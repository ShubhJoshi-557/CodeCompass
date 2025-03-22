import { CheckIcon, ClipboardIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "./button";

const CopyButton = ({ copyToClipboard }: any) => {
  const [hasCopied, setHasCopied] = useState(false);

  useEffect(() => {
    setTimeout(() => {
      setHasCopied(false);
    }, 2000);
  }, [hasCopied]);

  return (
    <Button
      size="icon"
      variant="ghost"
      className="relative z-10 h-6 w-6 cursor-pointer"
      onClick={() => {
        setHasCopied(true);
        copyToClipboard();
      }}
    >
      {hasCopied ? (
        <CheckIcon className="h-3 w-3" />
      ) : (
        <ClipboardIcon className="h-3 w-3" />
      )}
      <span className="sr-only">Copy</span>
    </Button>
  );
};

export default CopyButton;
