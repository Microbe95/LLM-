// utils/stepTracker.ts
const stepRoutes: Record<string, number> = {
  "/issue/project": 1,
  "/survey/1": 2,
  "/evaluate/1": 3,
  "/mapping/1": 4,
  "/report/1": 5,
};

export function markStepCompleteAuto() {
  if (typeof window === "undefined") return; // SSR 방지

  const user = localStorage.getItem("sessionUser");
  const allProjects = JSON.parse(localStorage.getItem("projects") || "{}");
  const currentId = localStorage.getItem("currentProjectId");
  const pathname = window.location.pathname;

  const stepNum = stepRoutes[pathname];
  if (!user || !currentId || !stepNum || !allProjects[user]) return;

  allProjects[user] = allProjects[user].map((p: any) =>
    p.id === currentId ? { ...p, [`step${stepNum}`]: true } : p
  );

  localStorage.setItem("projects", JSON.stringify(allProjects));
}