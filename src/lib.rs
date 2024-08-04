//! A Safe, familiar, embeddable programming language.
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
pub use muse_lang::*;
#[cfg(feature = "reactor")]
pub use muse_reactor as reactor;
#[cfg(feature = "ui")]
pub use muse_ui as ui;
