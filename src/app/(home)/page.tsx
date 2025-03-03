import SearchBar from "@/components/searchbar/SearchBar";

const page = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <div className="text-center text-3xl m-5">Hi, I'm CodeCompass. 👋</div>
      <div className="text-center text-sm mb-5">How can I assist you today?</div>
      <SearchBar />
    </div>
  );
};

export default page;
