import { useState } from "react";
import { FaFile, FaFolder, FaFolderOpen } from "react-icons/fa";

const buildFileTree = (files:any) => {
  let root = {};
  files.forEach((file:any) => {
    let parts = file.path.split("/");
    let current = root;

    for (let i = 0; i < parts.length; i++) {
      let part = parts[i];
      if (!current[part]) {
        current[part] = {
          name: part,
          type: i === parts.length - 1 ? file.type : "tree",
          children: {},
          url: file.url,
        };
      }
      current = current[part].children;
    }
  });

  return root;
};

const FileTreeNode = ({ node }:any) => {
  const [isOpen, setIsOpen] = useState(false);
  const hasChildren = node.children && Object.keys(node.children).length > 0;
  return (
    <div className="ml-0">
      {node.type === "tree" ? (
        <div
          onClick={() => setIsOpen(!isOpen)}
          className="cursor-pointer flex items-centerhover:bg-zinc-400 dark:hover:hover:bg-zinc-700 rounded-sm"
        >
          {isOpen ? (
            <FaFolderOpen className="mr-1 text-yellow-500" />
          ) : (
            <FaFolder className="mr-1 text-yellow-500" />
          )}
          {/* {isOpen ? <FolderOpen /> : <Folder />} */}
          {node.name}
        </div>
      ) : (
        <a
          href={node.url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center hover:bg-zinc-400 dark:hover:bg-zinc-700 rounded-sm"
        >
          <FaFile className="mr-1 text-gray-400" />
          {/* <File /> */}
          {node.name}
        </a>
      )}
      {hasChildren && isOpen && (
        <div className="ml-1 border-l pl-2">
          {Object.values(node.children).map((child:any) => (
            <FileTreeNode key={child.name} node={child} />
          ))}
        </div>
      )}
    </div>
  );
};

const FileTree = ({ data }:any) => {
  console.log(data, "DATA");
  const tree = buildFileTree(data);

  return (
    <div className="p-4 bg-neutral-200 dark:bg-neutral-800 rounded-lg shadow-md">
      {Object.values(tree).map((node:any) => (
        <FileTreeNode key={node.name} node={node} />
      ))}
    </div>
  );
};

export default FileTree;
