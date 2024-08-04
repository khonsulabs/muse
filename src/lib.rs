//! A Safe, familiar, embeddable programming language.

pub use muse_lang::*;

#[cfg(feature = "reactor")]
pub use muse_reactor as reactor;
#[cfg(feature = "ui")]
pub use muse_ui as ui;
