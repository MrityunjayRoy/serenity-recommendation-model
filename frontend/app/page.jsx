"use client";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Sparkles, ArrowRight } from "lucide-react";

export default function Home() {
  const router = useRouter();

  return (
    /* Flex wrapper to keep footer at the bottom */
    <div className="flex flex-col min-h-screen bg-white dark:bg-[#0a0a0a] text-black dark:text-zinc-100 transition-colors duration-300">

      {/* 🎵 Simple Header */}
      <header className="px-6 md:px-16 py-8 border-b border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto">
          <motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-2xl md:text-3xl font-extrabold tracking-tight"
          >
            Serenity<span className="text-[#0097b2]">.</span>
          </motion.h1>
        </div>
      </header>

      {/* 🌈 Main Hero Section */}
      <main className="flex-grow flex flex-col items-center justify-center px-6 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="max-w-2xl"
        >
          <h2 className="text-5xl md:text-7xl font-black mb-6 tracking-tight">
            Music for every <br />
            <span className="text-[#0097b2]">emotion.</span>
          </h2>

          <p className="text-zinc-500 dark:text-zinc-400 text-lg md:text-xl mb-10 max-w-lg mx-auto leading-relaxed">
            Experience a personalized journey through sound tailored to your current mood and vibe.
          </p>

          {/* The Main Call to Action */}
          <motion.button
            onClick={() => router.push('/recommend')}
            whileHover={{ scale: 1.02, boxShadow: "0 0 20px rgba(0, 151, 178, 0.4)" }}
            whileTap={{ scale: 0.95 }}
            className="group flex items-center gap-3 bg-[#0097b2] text-white px-8 py-4 rounded-2xl text-lg font-bold transition-all shadow-lg"
          >
            Go to Recommendation Page
            <ArrowRight className="group-hover:translate-x-1 transition-transform" />
          </motion.button>
        </motion.div>
      </main>

      {/* 🌙 Footer - Pushed to the bottom */}
      <footer className="w-full py-10 px-6 border-t border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            <p className="font-bold text-[#0097b2] text-lg mb-1">Serenity</p>
            <p className="text-zinc-500 dark:text-zinc-400 text-sm">
              Tailoring your sonic experience since 2024
            </p>
            <div className="mt-4 flex justify-center gap-4 text-xs font-medium text-zinc-400 dark:text-zinc-500">
              <p>Made with ❤️</p>
              <span>•</span>
              <a href="https://github.com/MrityunjayRoy/serenity" className="hover:text-[#0097b2] transition-colors">Github</a>
            </div>
          </motion.div>
        </div>
      </footer>
    </div>
  );
}