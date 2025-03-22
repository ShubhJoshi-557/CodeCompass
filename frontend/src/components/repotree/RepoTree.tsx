"use client";

import useStore from "@/store/store";
import "react-folder-tree/dist/style.css";
import { Skeleton } from "../ui/skeleton";
import FileTree from "./FileExplorer";

const RepoTree: React.FC = () => {
  const { currentRepo, repoTreeLoading } = useStore();
  if (repoTreeLoading)
    return (
      <div className="p-4  bg-neutral-200 dark:bg-neutral-800 rounded-lg shadow-md">
        <div className="space-y-2">
          <Skeleton className="h-4 w-[150px]" />
          <Skeleton className="ml-14 h-4 w-[150px]" />
          <Skeleton className="ml-14 h-4 w-[150px]" />
          <Skeleton className="h-4 w-[150px]" />
          <Skeleton className="ml-14 h-4 w-[150px]" />
          <Skeleton className="ml-14 h-4 w-[150px]" />
        </div>
      </div>
    );
  return <div>{!repoTreeLoading && currentRepo?.repoTree && <FileTree data={currentRepo?.repoTree} />}</div>;
};

export default RepoTree;
