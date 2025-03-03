import { SearchCode } from "lucide-react";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";

const SearchBar = () => {
  return (
    <div className="w-3xl max-h-[200px] mx-auto flex">
      <div className="w-full p-2 border-none rounded-3xl bg-neutral-200 dark:bg-neutral-600">
        <Textarea
          className="border-none outline-none resize-none max-h-[180px]"
          placeholder="Search code, understand logic, and debug smarter..."
          
        ></Textarea>
      </div>
      {/* <Input
        type="email"
        
        
      ></Input> */}
      <Button className="my-auto mx-2 cursor-pointer">
        Search <SearchCode />
      </Button>
    </div>
  );
};

export default SearchBar;
