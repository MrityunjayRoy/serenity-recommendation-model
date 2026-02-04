"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Sun, CloudRain, Waves, Flame, Headphones } from "lucide-react";

export default function Recommend() {
    const router = useRouter();
    const [userName, setUserName] = useState("Music Lover");

    const moodList = [
        { id: 1, name: "Happy", Icon: Sun, gradient: "from-yellow-400 via-amber-500 to-orange-500" },
        { id: 2, name: "Sad", Icon: CloudRain, gradient: "from-blue-600 via-indigo-700 to-slate-800" },
        { id: 3, name: "Calm", Icon: Waves, gradient: "from-emerald-400 via-teal-500 to-cyan-600" },
        { id: 4, name: "Angry", Icon: Flame, gradient: "from-rose-600 via-red-700 to-orange-700" },
    ];

    return (
        /* Flex wrapper ensures the footer stays at the bottom */
        <div className="flex flex-col min-h-screen bg-white dark:bg-[#0a0a0a] text-black dark:text-zinc-100 transition-colors duration-300">

            {/* 🎵 Header - Clean & Sticky-ready */}
            <header className="px-6 md:px-16 py-8 border-b border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/50 backdrop-blur-md sticky top-0 z-50">
                <div className="max-w-7xl mx-auto flex justify-between items-center">
                    <motion.h1
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="text-2xl md:text-3xl font-extrabold tracking-tight"
                    >
                        Serenity<span className="text-[#0097b2]">.</span>
                    </motion.h1>
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="text-sm md:text-base font-medium"
                    >
                        Welcome, <span className="text-[#0097b2]">{userName}</span>
                    </motion.div>
                </div>
            </header>

            {/* 🌈 Main Content - flex-grow pushes footer down */}
            <main className="flex-grow flex flex-col justify-center px-6 md:px-16 py-12">
                <section className="max-w-7xl mx-auto w-full text-center">
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mb-12"
                    >
                        <h2 className="text-4xl md:text-5xl font-bold mb-4 flex items-center justify-center gap-4">
                            How are you feeling? <Headphones className="text-[#0097b2]" size={40} />
                        </h2>
                        <p className="text-zinc-500 dark:text-zinc-400 text-lg">
                            Select a mood to start your personalized radio
                        </p>
                    </motion.div>

                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-8 max-w-5xl mx-auto">
                        {moodList.map(({ id, name, Icon, gradient }, index) => (
                            <motion.button
                                key={id}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.1 }}
                                onClick={() => router.push(`/recommend/${name.toLowerCase()}`)}
                                whileHover={{ scale: 1.05, y: -5 }}
                                whileTap={{ scale: 0.95 }}
                                className={`group relative aspect-square rounded-3xl overflow-hidden shadow-xl dark:shadow-black/50 bg-gradient-to-br ${gradient}`}
                            >
                                <div className="absolute inset-0 bg-black/10 group-hover:bg-transparent transition-colors" />
                                <div className="relative z-10 flex flex-col items-center justify-center h-full text-white">
                                    <Icon size={48} strokeWidth={2.5} className="mb-3 drop-shadow-md" />
                                    <span className="text-xl font-bold tracking-wider uppercase">{name}</span>
                                </div>
                            </motion.button>
                        ))}
                    </div>
                </section>
            </main>

            {/* 🌙 Footer - Always at the bottom */}
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
                            <a href="#" className="hover:text-[#0097b2] transition-colors">Github</a>
                        </div>
                    </motion.div>
                </div>
            </footer>
        </div>
    );
}