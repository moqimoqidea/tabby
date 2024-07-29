use std::sync::Arc;

use async_openai::config::OpenAIConfig;
use tabby_common::config::HttpModelConfig;
use tabby_inference::{ChatCompletionStream, ExtendedOpenAIConfig};
use tracing::{instrument, error, debug};

use crate::create_reqwest_client;

#[instrument(skip(model))]
pub async fn create(model: &HttpModelConfig) -> Arc<dyn ChatCompletionStream> {
    debug!("Creating OpenAIConfig with API base: {}", model.api_endpoint);
    let config = OpenAIConfig::default()
        .with_api_base(model.api_endpoint.clone())
        .with_api_key(model.api_key.clone().unwrap_or_default());

    let mut builder = ExtendedOpenAIConfig::builder();
    builder
        .base(config)
        .model_name(model.model_name.as_deref().expect("Model name is required"));

    if model.kind == "openai/chat" {
        debug!("Model kind is openai/chat, no fields to remove");
    } else if model.kind == "mistral/chat" {
        debug!("Model kind is mistral/chat, removing fields for mistral");
        builder.fields_to_remove(ExtendedOpenAIConfig::mistral_fields_to_remove());
    } else {
        error!("Unsupported model kind: {}", model.kind);
        panic!("Unsupported model kind: {}", model.kind);
    }

    let config = match builder.build() {
        Ok(cfg) => {
            debug!("Successfully built ExtendedOpenAIConfig");
            cfg
        },
        Err(e) => {
            error!("Failed to build ExtendedOpenAIConfig: {:?}", e);
            panic!("Failed to build config");
        }
    };

    Arc::new(
        async_openai::Client::with_config(config)
            .with_http_client(create_reqwest_client(&model.api_endpoint)),
    )
}
