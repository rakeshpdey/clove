/*
 * src/lib.rs
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Rakesh Pradip Dey
 *
 * Licensed under the MIT License <LICENSE-MIT or http://opensource.org/licenses/MIT>.
 * Note: Portions of this software are adapted from existing open-source frameworks.
 * This file may not be copied, modified, or distributed except according to the terms
 * of the MIT license.
 */

pub mod backend;
pub mod data;
pub mod device;
pub mod distributed;
pub mod ffi;
pub mod lazy;
pub mod nn;
pub mod optim;
pub mod tensor;

pub use device::EngineDevice;
pub use optim::AdamW;
pub use tensor::{Node, Tensor};
