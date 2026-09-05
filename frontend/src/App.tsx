import { useEffect, useState } from "react";
import { Layout } from "./components/Layout";
import { ErrorState, LoadingSkeleton } from "./components/ui";
import { useRequest } from "./hooks/useRequest";
import { api } from "./lib/api";
import type { ViewName } from "./types";
import { FixturesView } from "./views/FixturesView";
import { ModelView } from "./views/ModelView";
import { PlayersView } from "./views/PlayersView";

const views: ViewName[] = ["players", "fixtures", "model"];

function viewFromLocation(): ViewName {
  const value = window.location.hash.replace("#", "") as ViewName;
  return views.includes(value) ? value : "players";
}

export default function App() {
  const [activeView, setActiveView] = useState<ViewName>(viewFromLocation);
  const metadata = useRequest((signal) => api.metadata(signal), []);
  const health = useRequest((signal) => api.health(signal), []);

  useEffect(() => {
    const handleLocationChange = () => setActiveView(viewFromLocation());
    window.addEventListener("hashchange", handleLocationChange);
    return () => window.removeEventListener("hashchange", handleLocationChange);
  }, []);

  function navigate(view: ViewName) {
    window.location.hash = view;
    setActiveView(view);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <Layout
      activeView={activeView}
      onNavigate={navigate}
      metadata={metadata.data}
      healthy={health.data ? health.data.status === "ok" : health.error ? false : null}
    >
      {metadata.loading && !metadata.data ? (
        <LoadingSkeleton label="Loading inference metadata" variant={activeView === "fixtures" ? "cards" : "table"} withHeading />
      ) : metadata.error ? (
        <ErrorState error={metadata.error} retry={metadata.retry} />
      ) : metadata.data ? (
        <>
          {activeView === "players" && <PlayersView metadata={metadata.data} />}
          {activeView === "fixtures" && <FixturesView metadata={metadata.data} />}
          {activeView === "model" && <ModelView metadata={metadata.data} />}
        </>
      ) : null}
    </Layout>
  );
}
