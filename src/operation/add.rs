use super::{Kernel, Operation, ParamSpec, Result};

pub struct Add {}

impl Add {
    pub fn op() -> Box<dyn Operation> {
        Box::new(Self {})
    }
}

impl Operation for Add {
    fn name(&self) -> &str {
        "add"
    }

    fn kernel(&self, params: &ParamSpec) -> Result<std::sync::Arc<dyn Kernel>> {
        todo!()
    }
}
