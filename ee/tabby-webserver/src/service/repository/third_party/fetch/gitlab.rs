use gitlab::{
    api::{projects::Projects, AsyncQuery, Pagination},
    GitlabBuilder,
};
use serde::Deserialize;

use super::RepositoryInfo;

#[derive(Deserialize)]
pub struct GitlabRepository {
    pub id: u128,
    pub path_with_namespace: String,
    pub http_url_to_repo: String,
}

pub async fn fetch_all_gitlab_repos(
    access_token: &str,
    api_base: &str,
) -> Result<Vec<RepositoryInfo>, anyhow::Error> {
    // Gitlab client expects a url base like "gitlab.com" not "https://gitlab.com"
    // We still want to take a more consistent format as user input, so this
    // will help normalize it to prevent confusion
    let base_url = api_base.strip_prefix("https://").unwrap_or(api_base);
    let mut builder = GitlabBuilder::new(base_url, access_token);

    if cfg!(target_family = "unix") {
        let mut read_dir = tokio::fs::read_dir("/etc/ssl/certs").await?;
        while let Some(cert) = read_dir.next_entry().await? {
            let path = cert.path();
            if !path.ends_with(".pem") {
                continue;
            }
            let contents = tokio::fs::read(path).await?;
            builder.client_identity_from_pem(&contents);
        }
    }

    let gitlab = GitlabBuilder::new(base_url, access_token)
        .build_async()
        .await?;
    let repos: Vec<GitlabRepository> = gitlab::api::paged(
        Projects::builder().membership(true).build()?,
        Pagination::All,
    )
    .query_async(&gitlab)
    .await?;

    Ok(repos
        .into_iter()
        .map(|repo| RepositoryInfo {
            name: repo.path_with_namespace,
            git_url: repo.http_url_to_repo,
            vendor_id: repo.id.to_string(),
        })
        .collect())
}
