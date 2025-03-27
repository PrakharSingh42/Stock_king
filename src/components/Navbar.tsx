import Link from 'next/link';
import Image from 'next/image';

const Navbar = () => {
    return (
    
    <nav className="bg-gray-500 py-4 px-6 flex items-center justify-between shadow-md">
      {/* Logo (Left) */}
      <div className="flex items-center">
        <Link href="/" className="flex items-center space-x-2">
          <Image src="/logo.png" alt="Logo" width={40} height={40} />
          <span className="font-semibold text-xl">Your Brand</span>
        </Link>
      </div>
      {/* Navigation Links (Middle) */}
      <div className="flex space-x-4">
        <Link href="/portfolio" className="hover:text-gray-700">
          Portfolio
        </Link>
        <Link href="/prediction" className="hover:text-gray-700">
          Prediction
        </Link>
        <Link href="/buy-sell" className="hover:text-gray-700">
          Buy & Sell
        </Link>
      </div>
       {/* Profile Icon (Right) */}
       <div className="flex items-center">
        {/* Replace with your actual profile icon or component */}
        <button className="rounded-full bg-gray-300 w-8 h-8 flex items-center justify-center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"
              clipRule="evenodd"
            />
          </svg>
        </button>
        </div>

        </nav>
    );
};

export default Navbar;