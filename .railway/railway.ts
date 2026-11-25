import { defineRailway, github, project, service } from "railway/iac";
// This repository manages only its own resources in the environment. Other
// repositories export their own partial name.
// See https://docs.railway.com/infrastructure-as-code#multi-repo-projects
export const partial = "hn-search-web";
export default defineRailway(() => {
  const hnSearchWeb = service("hn-search-web", {
    source: github("afiodorov/hn-search", { branch: "main", checkSuites: false }),
    build: { buildEnvironment: "V3", builder: "RAILPACK", watchPatterns: ["Dockerfile", "uv.lock", ".python-version", "hn_search/**", "frontend/**"] },
    replicas: { "europe-west4-drams3a": 1 },
    domains: ["hn.fiodorov.es"],
  });

  return project("hn-search", {
    variables: { managed: false },
    resources: [hnSearchWeb],
  });
});
