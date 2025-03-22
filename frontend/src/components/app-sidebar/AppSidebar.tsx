"use client";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import useStore from "@/store/store";
import { Edit } from "lucide-react";
import Link from "next/link";
import RepoTree from "../repotree/RepoTree";
import { SelectRepo } from "../settings/SelectRepo";
import { Button } from "../ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { UserTab } from "./UserTab";

export function AppSidebar() {
  const { currentRepo, currentFolder } = useStore();
  const FolderPath = ({ path }: any) => {
    const getDisplayName = (path) => {
      const parts = path.split("/");
      if (parts.length === 1) return path; // Single folder case
      return `.../${parts[parts.length - 1]}`;
    };

    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            className="cursor-pointer text-blue-500"
            onClick={() => {
              document.getElementById(`${currentFolder}`)?.scrollIntoView({
                behavior: "smooth", block: "center" 
              });
            }}
          >
            {getDisplayName(path)}
          </TooltipTrigger>
          <TooltipContent>{path}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  };
  return (
    <Sidebar className="border-none">
      <Link href="/">
        <SidebarHeader className="m-auto p-5 text-2xl">
          {"CodeCompass </>"}
        </SidebarHeader>
      </Link>

      <SidebarContent>
        <SidebarMenu>
          <SidebarMenuItem className="px-3 pt-2">
            <Link href="/">
              <Button className="font-semibold">
                <Edit />
                New Chat
              </Button>
            </Link>
          </SidebarMenuItem>
        </SidebarMenu>

        <SidebarGroup>
          <SidebarGroupLabel>Repository</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SelectRepo />
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        {currentRepo?.link && currentRepo?.repo && currentRepo?.branch && (
          <SidebarGroup>
            <SidebarGroupLabel>Explorer</SidebarGroupLabel>
            <SidebarGroupContent>
              <div className="rounded-lg mb-2 py-2 px-5 text-[14px] bg-neutral-200 dark:bg-neutral-800 flex justify-between">
                <div>Current Folder: </div>
                <FolderPath path={currentFolder} />
                {/* !== "" ? "/" + (currentFolder.split("/").length>2) : ""} */}
              </div>
              <SidebarMenu>
                <SidebarMenuItem>
                  <RepoTree
                    owner={currentRepo?.owner}
                    repo={currentRepo?.repo}
                    branch={currentRepo?.branch}
                  />
                </SidebarMenuItem>
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
        <SidebarGroup>
          <SidebarGroupLabel>Recent Searches</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton>
                  <Link href="/q/dbqkiyhgkyg">DB Query</Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton>
                  <Link href="/q/apiqkiyhgkyg">API Request Query</Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton>
                  <Link href="/q/docqkiyhgkyg">Documentation Query</Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <UserTab />
    </Sidebar>
  );
}
