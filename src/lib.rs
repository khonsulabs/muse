//! A Safe, familiar, embeddable programming language.
#![cfg_attr(all(docsrs, not(doctest)), feature(doc_auto_cfg))]

pub use muse_lang::*;

#[cfg(feature = "channel")]
pub use muse_channel as channel;
#[cfg(feature = "reactor")]
pub use muse_reactor as reactor;
#[cfg(feature = "ui")]
pub use muse_ui as ui;
