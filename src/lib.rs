use std::collections::HashMap;

/// A buffer of signal samples.
pub type Buffer<const SIZE: usize> = [f32; SIZE];

/// Performs an isolated processing step in a signal graph.
pub trait Node<const BUFFER_SIZE: usize, const SAMPLE_RATE: usize> {
    /// Generate or process serial data.
    ///
    /// The `buffer` parameter serves as both input and output.
    fn generate(&mut self, buffer: &mut Buffer<BUFFER_SIZE>);
}

/// Internal wrapper for nodes to support graph operations.
struct NodeContainer<const B: usize, const S: usize> {
    key: usize,
    node: Box<dyn Node<B, S>>,
    dependencies: Vec<usize>,
    dependents: Vec<usize>,
}

/// Use this value in the `Graph::connect` method to indicate that a node receives external input.
pub const EXTERNAL_INPUT: usize = 0;

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
    /// This also removes all connections to and from this node. Returns true if the node was found
    /// and removed.
    pub fn remove_node(&mut self, key: usize) -> bool {
        let mut did_remove = false;
        self.nodes.retain(|n| {
            let matching = n.key != key;
            if !matching {
                did_remove = true;
            }
            matching
        });

        for node in &mut self.nodes {
            node.dependencies.retain(|&v| v != key);
            node.dependents.retain(|&v| v != key);
        }

        did_remove
    }

    /// Create a connection between a source node and a sink node.
    ///
    /// When signal is generated, the output of the source will be provided to the sink.
    pub fn connect(&mut self, source: usize, sink: usize) -> Result<(), ConnectError> {
        // no self connections or cycles allowed
        if source == sink {
            return Err(ConnectError::SelfReference);
        }
        if self.would_have_cycle(source, sink) {
            return Err(ConnectError::CyclicalReference);
        }

        match self.nodes.iter().enumerate().fold(
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
            (Some(source_index), Some(sink_index)) => {
                self.nodes[source_index].dependents.push(sink);
                self.nodes[sink_index].dependencies.push(source);
                if source_index > sink_index {
                    // bubble source node up the list without otherwise disturbing the order
                    for index in (sink_index..source_index).rev() {
                        self.nodes.swap(index, index + 1);
                    }
                }
            }
            (None, Some(sink_index)) if source == EXTERNAL_INPUT => {
                self.nodes[sink_index].dependencies.push(EXTERNAL_INPUT);
            }
            _ => (),
        }

        Ok(())
    }

    /// Check for a potential cycle in the graph, before creating one.
    fn would_have_cycle(&self, source: usize, sink: usize) -> bool {
        if source == EXTERNAL_INPUT {
            return false;
        }

        self.nodes
            .iter()
            .find(|NodeContainer { key, .. }| *key == source)
            .expect("Could not find source node when attempting to connect nodes")
            .dependencies
            .iter()
            .any(|&d| d == sink || self.would_have_cycle(d, sink))
    }

    /// Remove a connection between nodes.
    ///
    /// Returns true if a connection was found and removed.
    pub fn disconnect(&mut self, source: usize, sink: usize) -> bool {
        let mut did_disconnect = false;

        for node in &mut self.nodes {
            if node.key == source {
                node.dependents.retain(|&v| {
                    let no_match = v != sink;
                    if !no_match {
                        did_disconnect = true;
                    }
                    no_match
                });
            }
            if node.key == sink {
                node.dependencies.retain(|&v| {
                    let no_match = v != source;

                    if !no_match {
                        did_disconnect = true;
                    }

                    no_match
                });
            }
        }

        did_disconnect
    }

    /// Process the given buffer of input signal.
    ///
    /// The `buffer` parameter serves as both input and output.
    pub fn generate(&mut self, buffer: &mut Buffer<BUFFER_SIZE>) {
        let mut buffer_cache = HashMap::new();
        buffer_cache.insert(EXTERNAL_INPUT, buffer.to_owned());

        for node in &mut self.nodes {
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

        *buffer = [0.0; BUFFER_SIZE];
        sum_buffers(
            self.nodes
                .iter()
                // find terminal nodes
                .filter(|NodeContainer { dependents, .. }| dependents.is_empty())
                .map(|NodeContainer { key, .. }| {
                    buffer_cache
                        .get(key)
                        .expect("Cache miss - this indicates a logic error in agora::Graph")
                }),
            buffer,
        );
    }
}

#[derive(Debug, PartialEq)]
pub enum ConnectError {
    SelfReference,
    CyclicalReference,
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

#[cfg(test)]
mod node_management {
    use super::{Buffer, ConnectError, Graph, Node};

    struct NopNode;
    impl<const B: usize, const S: usize> Node<B, S> for NopNode {
        fn generate(&mut self, _buffer: &mut Buffer<B>) {}
    }

    #[test]
    fn can_insert_noop_node() {
        let mut g = Graph::<1, 1>::default();
        g.add_node(Box::new(NopNode));
    }

    #[test]
    fn can_remove_node() {
        let mut g = Graph::<1, 1>::default();
        let node_a = g.add_node(Box::new(NopNode));
        assert!(g.remove_node(node_a));
    }

    #[test]
    fn reports_removal_status() {
        let mut g = Graph::<1, 1>::default();
        assert!(!g.remove_node(4));
    }

    #[test]
    fn can_connect_nodes() -> Result<(), ConnectError> {
        let mut g = Graph::<1, 1>::default();
        let node_a = g.add_node(Box::new(NopNode));
        let node_b = g.add_node(Box::new(NopNode));
        g.connect(node_a, node_b)
    }

    #[test]
    fn can_detect_self_references() {
        let mut g = Graph::<1, 1>::default();
        let node_a = g.add_node(Box::new(NopNode));
        let connect_result = g.connect(node_a, node_a);
        assert!(connect_result.is_err());
        assert_eq!(connect_result.unwrap_err(), ConnectError::SelfReference);
    }

    #[test]
    fn can_detect_cyclical_references() {
        let mut g = Graph::<1, 1>::default();
        let node_a = g.add_node(Box::new(NopNode));
        let node_b = g.add_node(Box::new(NopNode));
        let node_c = g.add_node(Box::new(NopNode));
        assert!(g.connect(node_a, node_b).is_ok());
        assert!(g.connect(node_b, node_c).is_ok());
        let connect_result = g.connect(node_c, node_a);
        assert!(connect_result.is_err());
        assert_eq!(connect_result.unwrap_err(), ConnectError::CyclicalReference);
    }

    #[test]
    fn can_disconnect_nodes() {
        let mut g = Graph::<1, 1>::default();
        let node_a = g.add_node(Box::new(NopNode));
        let node_b = g.add_node(Box::new(NopNode));
        assert!(g.connect(node_a, node_b).is_ok());
        assert!(g.disconnect(node_a, node_b));
    }

    #[test]
    fn reports_disconnect_status() {
        let mut g = Graph::<1, 1>::default();
        let node_a = g.add_node(Box::new(NopNode));
        let node_b = g.add_node(Box::new(NopNode));
        assert!(!g.disconnect(node_a, node_b));
    }
}

#[cfg(test)]
mod signal_processing {
    use super::{Buffer, Graph, Node, EXTERNAL_INPUT};
    use std::f32::consts::{PI, TAU};

    struct DcOffsetNode(f32);
    impl<const B: usize, const S: usize> Node<B, S> for DcOffsetNode {
        fn generate(&mut self, buffer: &mut Buffer<B>) {
            buffer.iter_mut().for_each(|v| *v += self.0);
        }
    }

    #[test]
    fn generates_basic_signal() {
        let mut g = Graph::<20, 1>::default();
        g.add_node(Box::new(DcOffsetNode(1.0)));
        let mut buffer = [0.0; 20];
        g.generate(&mut buffer);
        assert_eq!(buffer, [1.0; 20]);
    }

    #[test]
    fn processes_external_signal() {
        let mut g = Graph::<20, 1>::default();
        let node_a = g.add_node(Box::new(DcOffsetNode(-2.0)));
        g.connect(EXTERNAL_INPUT, node_a).unwrap();
        let mut buffer = [1.0; 20];
        g.generate(&mut buffer);
        assert_eq!(buffer, [-1.0; 20]);
    }

    struct PulseNode {
        frequency: f32,
        current_phase: f32,
    }
    impl<const B: usize, const S: usize> Node<B, S> for PulseNode {
        fn generate(&mut self, buffer: &mut Buffer<B>) {
            buffer.fill_with(|| {
                let last_phase = self.current_phase;
                self.current_phase += TAU / (self.frequency * S as f32);
                self.current_phase %= TAU;
                if last_phase < PI {
                    1.0
                } else {
                    0.0
                }
            });
        }
    }

    #[test]
    fn generates_wave_statefully() {
        let mut g = Graph::<20, 2>::default();
        g.add_node(Box::new(PulseNode {
            frequency: 1.0,
            current_phase: 0.0,
        }));
        let mut buffer = [0.0; 20];
        g.generate(&mut buffer);
        let expected = [
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
        ];
        assert_eq!(buffer, expected);
    }
}
