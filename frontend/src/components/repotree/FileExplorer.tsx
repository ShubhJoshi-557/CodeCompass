import useStore from "@/store/store";
import { useState } from "react";
import { FaFile, FaFolder, FaFolderOpen } from "react-icons/fa";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";

const buildFileTree = (files: any) => {
  const root: any = {};

  files.forEach((file: any) => {
    const parts = file.path.split("/");
    let current = root;
    let fullPath = "";

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      fullPath += (i === 0 ? "" : "/") + part;

      if (!current[part]) {
        current[part] = {
          name: part,
          type: i === parts.length - 1 ? file.type : "tree",
          children: {},
          url: file.url,
          path: fullPath, // Adding full file path
        };
      }

      current = current[part].children;
    }
  });

  return root;
};

const FileTreeNode = ({ node }: any) => {
  const [isOpen, setIsOpen] = useState(false);
  const { currentFolder, setCurrentFolder } = useStore();
  const hasChildren = node.children && Object.keys(node.children).length > 0;
  function truncateString(str: string) {
    return str.length > 15 ? str.slice(0, 15) + "..." : str;
  }
  return (
    <div className="ml-0">
      {node.type === "tree" ? (
        <div
          id={node.path}
          onClick={() => {
            setIsOpen(!isOpen);
            setCurrentFolder(node.path);
          }}
          className={`cursor-pointer flex items-center hover:bg-zinc-400 dark:hover:bg-zinc-700 rounded-sm ${
            node.path === currentFolder && "bg-zinc-400 dark:bg-zinc-700"
          }`}
        >
          {isOpen ? (
            <FaFolderOpen className="mx-1 my-auto text-yellow-500" />
          ) : (
            <FaFolder className="mx-1 my-auto text-yellow-500" />
          )}
          {/* {isOpen ? <FolderOpen /> : <Folder />} */}
          <div>{node.name}</div>
        </div>
      ) : (
        <div className="flex cursor-pointer items-center hover:bg-zinc-400 dark:hover:bg-zinc-700 rounded-sm">
          <FaFile className="mx-1 my-auto text-gray-400" />
          {/* <File /> */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <div className="cursor-pointer">
                  {truncateString(node.name)}
                </div>
              </TooltipTrigger>
              <TooltipContent>{node.path}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}
      {hasChildren && isOpen && (
        <div className="ml-1 border-l pl-2">
          {Object.values(node.children).map((child: any) => (
            <FileTreeNode key={child.name} node={child} />
          ))}
        </div>
      )}
    </div>
  );
};

const FileTree = ({ data }: any) => {
  const tree = buildFileTree(data);
  return (
    <div className="p-4 bg-neutral-200 dark:bg-neutral-800 rounded-lg shadow-md">
      {Object.values(tree).map((node: any) => (
        <FileTreeNode key={node.name} node={node} />
      ))}
    </div>
  );
};

export default FileTree;
