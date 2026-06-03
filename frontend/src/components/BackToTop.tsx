import { useEffect, useState } from "react";
import { ArrowUp } from "lucide-react";

interface Props {
  /** Container element to scroll. Defaults to window. */
  scrollEl?: HTMLElement | null;
  threshold?: number;
}

export default function BackToTop({ scrollEl, threshold = 600 }: Props) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el: Window | HTMLElement = scrollEl ?? window;
    const get = () => (scrollEl ? scrollEl.scrollTop : window.scrollY);
    const onScroll = () => setVisible(get() > threshold);
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [scrollEl, threshold]);

  if (!visible) return null;

  return (
    <button
      onClick={() => {
        if (scrollEl) {
          scrollEl.scrollTo({ top: 0, behavior: "smooth" });
        } else {
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
      }}
      className="fixed bottom-6 right-6 z-40 flex items-center gap-1.5 rounded-full border border-bg-border bg-bg-panel/90 px-3 py-2 text-xs font-medium text-slate-100 shadow-2xl backdrop-blur hover:border-studio-data hover:text-studio-data"
      title="Back to top"
    >
      <ArrowUp size={14} /> Top
    </button>
  );
}
