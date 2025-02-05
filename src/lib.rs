use bytemuck::NoUninit;
use core::fmt;
use generic_array::{ArrayLength, GenericArray};
use std::collections::HashSet;
use std::hash::Hash;
use std::{any::TypeId, marker::PhantomData, sync::Arc};
use wgpu::util::DeviceExt;

struct Library {
    operations: HashSet<Box<dyn Operation>>,
}

/// An [`Operation`] is an abstract function which can apply to an arbitrary number of parameters of varying type.
///
/// The [`Operation`] is not itself a unique kernel, but instead is used to generate a kernel based on
trait Operation {
    /// Returns the unique name of the operation.
    fn name(&self) -> &str;

    /// Returns a kernel specialized to the input types and dimensions, if possible.
    ///
    /// Returns `None` if this specialization is not allowed.
    fn kernel(&self, params: &ParamSpec) -> Option<Arc<dyn Kernel>>;
}

impl Hash for dyn Operation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

/// A [`Kernel`] is a concrete shader implementation with a specific entry point name.
///
/// The [`fmt::Display`] trait is used to write the kernel to a stream or convert it to a string.
trait Kernel: fmt::Display {
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

/// [`TensorDynamic`] works the same as [`TensorDense`], but type parameters are erased.
///
/// Internally this remembers the concrete type using [`TypeId`].
pub struct TensorDynamic {
    buffer: wgpu::Buffer,
    dims: Vec<u32>,
    ty: TypeId,
}

/// [`TensorAbstract`] represents a node in a computation DAG.
///
/// The [`TensorAbstract`] must first be converted to a [`TensorOutput`] before execution to get a [`TensorDense`].
pub struct TensorAbstract<T, D: ArrayLength> {
    _phantom: PhantomData<(T, D)>,
}

/// [`TensorInput`] is an input node to a computation DAG.
///
/// You can convert this into [`TensorAbstract`] to perform computations.
pub struct TensorInput<T, D: ArrayLength> {
    _phantom: PhantomData<(T, D)>,
}

/// [`TensorOutput`] is an output node to a computation DAG.
///
/// You can convert a [`TensorAbstract`] into this before execution to get output data as a [`TensorDense`].
pub struct TensorOutput<T, D: ArrayLength> {
    _phantom: PhantomData<(T, D)>,
}

/// [`TensorDense`] stores data with the largest memory strides first.
///
/// The generic argument `D` specifies the dimensionality of the tensor.
pub struct TensorDense<T, D: ArrayLength> {
    buffer: wgpu::Buffer,
    dims: GenericArray<u32, D>,
    _phantom: PhantomData<T>,
}

impl<T, D: ArrayLength> TensorDense<T, D>
where
    T: NoUninit,
{
    pub fn new(device: &wgpu::Device, data: &[T], dims: impl Into<GenericArray<u32, D>>) -> Self {
        // Create the data buffer.
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // Note that the dims here stay on the CPU.
        let dims = dims.into();

        Self {
            buffer,
            dims,
            _phantom: PhantomData,
        }
    }
}

pub enum Primitive {
    I32,
    U32,
    F32,
}

pub struct TensorSparse {
    buffer: wgpu::Buffer,
    strides: wgpu::Buffer,
    dims: wgpu::Buffer,
}

impl TensorSparse {
    pub fn new(device: &wgpu::Device, data: &[u8], strides: &[i32], dims: &[u32]) -> Self {
        // TODO: This doesn't handle strides correctly because the starting index for striding isn't supplied.

        // We can't have an index that wont fit in a u32.
        assert!(data.len() < (1 << 32));
        // Validate that the strides and dims are of the same count.
        assert_eq!(strides.len(), dims.len());
        // Compute the actual length of the data as per the strides.
        let accessible_len = strides
            .iter()
            .zip(dims.iter())
            .map(|(&stride, &dim)| {
                stride
                    .checked_abs()
                    .and_then(|stride| (stride as u32).checked_mul(dim))
            })
            .reduce(|a, b| {
                let a = a?;
                let b = b?;
                a.checked_add(b)
            })
            .unwrap_or(Some(1))
            .expect("overflow in tensor index with given strides");
        assert!(accessible_len <= data.len() as u32);

        let _ = device;
        todo!();
    }
}

pub async fn run() {
    let mut local_a = [0i32; 100];
    for (i, e) in local_a.iter_mut().enumerate() {
        *e = i as i32;
    }
    log::info!("Input a: {local_a:?}");
    let mut local_b = [0i32; 100];
    for (i, e) in local_b.iter_mut().enumerate() {
        *e = i as i32 * 2;
    }
    log::info!("Input b: {local_b:?}");

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .unwrap();

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let storage_buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&local_a[..]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let storage_buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&local_b[..]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_of_val(&local_a) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage_buffer_b.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    //----------------------------------------------------------

    let mut command_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        /* Note that since each workgroup will cover both arrays, we only need to
        cover the length of one array. */
        compute_pass.dispatch_workgroups(local_a.len() as u32, 1, 1);
    }
    queue.submit(Some(command_encoder.finish()));

    //----------------------------------------------------------

    get_data(
        &mut local_a[..],
        &storage_buffer_a,
        &output_staging_buffer,
        &device,
        &queue,
    )
    .await;
    get_data(
        &mut local_b[..],
        &storage_buffer_b,
        &output_staging_buffer,
        &device,
        &queue,
    )
    .await;

    log::info!("Output in A: {local_a:?}");
    log::info!("Output in B: {local_b:?}");
}

async fn get_data<T: bytemuck::Pod>(
    output: &mut [T],
    storage_buffer: &wgpu::Buffer,
    staging_buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let mut command_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    command_encoder.copy_buffer_to_buffer(
        storage_buffer,
        0,
        staging_buffer,
        0,
        size_of_val(output) as u64,
    );
    queue.submit(Some(command_encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    receiver.recv_async().await.unwrap().unwrap();
    output.copy_from_slice(bytemuck::cast_slice(&buffer_slice.get_mapped_range()[..]));
    staging_buffer.unmap();
}
