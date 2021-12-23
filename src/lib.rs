use std::collections::HashMap;

/// A buffer of signal samples.
pub type Buffer<const SIZE: usize> = [f32; SIZE];

/// Performs an isolated processing step in a signal graph.
pub trait Node<const BUFFER_SIZE: usize, const SAMPLE_RATE: usize> {
    /// Generate or process serial data.
    ///
    /// The `buffer` parameter serves as both input and output.
    fn generate(&self, buffer: &mut Buffer<BUFFER_SIZE>);
}

/// Internal wrapper for nodes to support graph operations.
struct NodeContainer<const B: usize, const S: usize> {
    key: usize,
    node: Box<dyn Node<B, S>>,
    dependencies: Vec<usize>,
    dependents: Vec<usize>,
}

const EXTERNAL_INPUT: usize = 0;

/// Orchestrate a group of processing nodes to produce sound.
///
/// Sample rate and buffer length (in samples) must be constant, but the nodes and connections can
/// be configured at runtime.
#[derive(Default)]
pub struct Graph<const BUFFER_SIZE: usize, const SAMPLE_RATE: usize> {
    current_key: usize,
    nodes: Vec<NodeContainer<BUFFER_SIZE, SAMPLE_RATE>>,
}

impl<const BUFFER_SIZE: usize, const SAMPLE_RATE: usize> Graph<BUFFER_SIZE, SAMPLE_RATE> {
    /// Inserts a node into the signal graph, returning its ID.
    ///
    /// Note that simply adding a node does not connect it to any other nodes. The key returned
    /// from this method can be passed into the `remove`, `connect`, and `disconnect` methods.
    pub fn add_node(&mut self, node: Box<dyn Node<BUFFER_SIZE, SAMPLE_RATE>>) -> usize {
        // this will skip zero, intentionally. in the `generate` step, zero will represent
        // _external_ signal input.
        self.current_key += 1;
        self.nodes.push(NodeContainer {
            key: self.current_key,
            node,
            dependencies: Vec::new(),
            dependents: Vec::new(),
        });
        self.current_key
    }

    /// Removes a node from the signal graph.
    ///
    /// This also removes all connections to and from this node.
    pub fn remove_node(&mut self, key: usize) {
        self.nodes.retain(|n| n.key != key);
        for node in &mut self.nodes {
            node.dependencies.retain(|&v| v != key);
            node.dependents.retain(|&v| v != key);
        }
    }

    /// Create a connection between a source node and a sink node.
    ///
    /// When signal is generated, the output of the source will be provided to the sink.
    pub fn connect(&mut self, source: usize, sink: usize) {
        // no self connections or cycles allowed
        if source == sink || self.would_have_cycle(source, sink) {
            return;
        }

        if let (Some(source_index), Some(sink_index)) = self.nodes.iter().enumerate().fold(
            (None, None),
            |(maybe_source, maybe_sink), (index, NodeContainer { key, .. })| {
                if *key == source {
                    (Some(index), maybe_sink)
                } else if *key == sink {
                    (maybe_source, Some(index))
                } else {
                    (maybe_source, maybe_sink)
                }
            },
        ) {
            self.nodes[source_index].dependents.push(sink);
            self.nodes[sink_index].dependencies.push(source);
            if source_index > sink_index {
                // bubble source node up the list without otherwise disturbing the order
                for index in (sink_index..source_index).rev() {
                    self.nodes.swap(index, index + 1);
                }
            }
        }
    }

    /// Check for a potential cycle in the graph, before creating one.
    fn would_have_cycle(&self, source: usize, sink: usize) -> bool {
        self.nodes
            .iter()
            .find(|NodeContainer { key, .. }| *key == sink)
            .map(|NodeContainer { dependencies, .. }| {
                dependencies.iter().any(|&d| d == source || self.would_have_cycle(source, d))
            })
            .unwrap_or(false)
    }

    /// Remove a connection between nodes.
    pub fn disconnect(&mut self, source: usize, sink: usize) {
        for node in &mut self.nodes {
            if node.key == source {
                node.dependents.retain(|&v| v != sink);
            }
            if node.key == sink {
                node.dependencies.retain(|&v| v != source);
            }
        }
    }

    /// Process the given buffer of input signal.
    ///
    /// The `buffer` parameter serves as both input and output.
    pub fn generate(&self, buffer: &mut Buffer<BUFFER_SIZE>) {
        let mut buffer_cache = HashMap::new();
        buffer_cache.insert(EXTERNAL_INPUT, buffer.to_owned());

        for node in &self.nodes {
            let mut buf = [0.0; BUFFER_SIZE];
            sum_buffers(
                node.dependencies.iter().map(|i| {
                    buffer_cache
                        .get(i)
                        .expect("Cache miss - this indicates a logic error in agora::Graph")
                }),
                &mut buf,
            );
            node.node.generate(&mut buf);
            buffer_cache.insert(node.key, buf);
        }

        let terminal_nodes: Vec<_> = self
            .nodes
            .iter()
            .filter(|NodeContainer { dependents, .. }| dependents.is_empty())
            .map(|NodeContainer { key, .. }| *key)
            .collect();

        *buffer = [0.0; BUFFER_SIZE];
        sum_buffers(
            terminal_nodes.iter().map(|i| {
                buffer_cache
                    .get(i)
                    .expect("Cache miss - this indicates a logic error in agora::Graph")
            }),
            buffer,
        );
    }
}

/// Sum all inputs into a buffer.
fn sum_buffers<'a, I, const SIZE: usize>(inputs: I, buffer: &mut Buffer<SIZE>)
where
    I: IntoIterator<Item = &'a Buffer<SIZE>>,
{
    let mut num_inputs: usize = 0;

    for input in inputs {
        for (sample, dependency_sample) in buffer.iter_mut().zip(input) {
            *sample += dependency_sample;
        }
        num_inputs += 1;
    }

    if num_inputs > 1 {
        buffer
            .iter_mut()
            .for_each(|sample| *sample /= num_inputs as f32);
    }
}
