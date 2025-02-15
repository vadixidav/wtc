mod add;

use std::{any::TypeId, borrow::Borrow, fmt, hash::Hash, result, sync::Arc};

pub use add::*;

pub type Result<T> = result::Result<T, &'static str>;

/// An [`Operation`] is an abstract function which can apply to an arbitrary number of parameters of varying type.
///
/// The [`Operation`] is not itself a unique kernel, but instead is used to generate a kernel based on
pub(crate) trait Operation {
    /// Returns the unique name of the operation.
    fn name(&self) -> &str;

    /// Returns a kernel specialized to the input types and dimensions, if possible.
    ///
    /// Returns an error if this specialization is not allowed.
    fn kernel(&self, params: &ParamSpec) -> Result<Arc<dyn Kernel>>;
}

impl Borrow<str> for dyn Operation {
    fn borrow(&self) -> &str {
        self.name()
    }
}

impl Hash for dyn Operation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl PartialEq for dyn Operation {
    fn eq(&self, other: &Self) -> bool {
        self.name().eq(other.name())
    }
}

impl Eq for dyn Operation {}

/// A [`Kernel`] is a concrete shader implementation with a specific entry point name.
///
/// The [`fmt::Display`] trait is used to write the kernel to a stream or convert it to a string.
pub(crate) trait Kernel: fmt::Display {
    /// Returns the unique name of the kernel operation entry point.
    fn name(&self) -> &str;

    /// Returns a list of dependencies required by this kernel.
    fn dependencies(&self) -> Vec<CallSpec>;
}

/// A specific specialization of an operation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallSpec {
    /// The name of the operation being called.
    operation: String,
    /// The parameter spec for the operation call.
    params: ParamSpec,
}

/// A specific set of parameters.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParamSpec {
    accumulators: Vec<TensorSpec>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

/// A specific type and dimension count required for kernel specialization.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorSpec {
    dims: u32,
    ty: TypeId,
}
