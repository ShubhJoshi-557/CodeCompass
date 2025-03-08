import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import useStore from "@/store/store";
import { Folder } from "lucide-react";
import { ProfileForm } from "./Form";

export function SelectRepo() {
  const { currentRepo, updateCurrentRepo } = useStore();
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="secondary"
          className="w-full border  flex justify-between"
        >
          {currentRepo?.link === "" && (<span className="my-auto">Select Repo</span>)}
          {currentRepo?.link &&
            currentRepo?.owner &&
            currentRepo?.repo &&
            currentRepo?.branch && (
              <div className="flex justify-between w-full">
                <p>
                  {currentRepo?.owner}/{currentRepo?.repo}
                </p>
                <p>{`(${currentRepo?.branch})`}</p>
              </div>
            )}

          <Folder className="w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Select Repository</DialogTitle>
          <DialogDescription>
            Enter the Repository link, Branch name and Personal Access Token (If
            Applicable).
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <ProfileForm />
        </div>
        {/* <DialogFooter>
          <Button type="submit">Save changes</Button>
        </DialogFooter> */}
      </DialogContent>
    </Dialog>
  );
}