//! A library for managing signal processing networks.
//!
//! The main concept behind this library is that individual processing steps will be implemented as
//! [Nodes](Node) which are arranged into a [Graph]. Nodes can receive input signals, both internal
//! and external, and are expected to produce an output signal.
//!
//! Once you create and [register](Graph::add_node) a Node with a Graph instance, you can
//! [connect](Graph::connect) any two Nodes in an arbitrary manner. You can configure any Node to
//! receive [external input](EXTERNAL_INPUT) here as well. The connections and set of Nodes can be
//! modified at any time, and changes will take effect the next time you [process a
//! buffer](Graph::process).
//!
//! ### Assumptions and caveats
//!
//! - This library is built to work with double-precision floating point signals.
//! - Nodes can only consume one stream of input data and produce one stream of output data.
//! - Signal graphs must be acyclic - no feedback loops are permitted.
//! - Buffer length and sample rate must be constant (per Graph and Node instance) at compile time.
//! The tradeoff here is between safety (via the type system) versus flexibility, and for this
//! library safety is preferred.
//!
//! ### Resource usage
//!
//! - The time complexity of processing a buffer of data is linear in the number of nodes, and
//! memory usage is linear in both the number of nodes and the specified buffer size.
//! - The time complexity of adding a node is constant (amortized).
//! - The time complexity of removing a node is linear in the number of nodes.
//! - The time complexity of connecting two nodes is exponential in the number of nodes. This is
//! required in order to prevent cycles in the graph.
//! - The time complexity of disconnecting two notes is constant (amortized).
//!
//! ### Example
//! ```
//! # use agora::{Buffer, Node, Graph};
//! # const BUF_LEN: usize = 320;
//! struct DcOffsetNode { offset: f64 }
//! impl Node<BUF_LEN, 44_100> for DcOffsetNode {
//!     fn process(&mut self, buffer: &mut Buffer<BUF_LEN>) {
//!         for sample in buffer { *sample += self.offset }
//!     }
//! }
//!
//! let mut graph = Graph::default();
//! let node_0 = graph.add_node(Box::new(DcOffsetNode { offset: 1.0 }));
//!
//! let mut buffer = [0.0; BUF_LEN];
//! graph.process(&mut buffer);
//! assert_eq!(buffer, [1.0; BUF_LEN]);
//!
//! let node_1 = graph.add_node(Box::new(DcOffsetNode { offset: 2.0 }));
//! graph.connect(node_0, node_1);
//!
//! buffer = [0.0; BUF_LEN];
//! graph.process(&mut buffer);
//! assert_eq!(buffer, [3.0; BUF_LEN]);
//! ```

#![warn(missing_docs)]

use std::collections::{HashMap, HashSet};

/// A buffer of signal samples.
pub type Buffer<const SIZE: usize> = [f64; SIZE];

/// Performs an isolated processing step in a signal graph.
pub trait Node<const BUFFER_SIZE: usize, const SAMPLE_RATE: usize> {
    /// Generate or process serial data.
    ///
    /// The `buffer` parameter serves as both input and output.
    fn process(&mut self, buffer: &mut Buffer<BUFFER_SIZE>);
}

/// Provides a reference to a [Node] added to a [Graph].
///
/// Returned by [Graph::add_node], for later use with the [Graph::remove_node], [Graph::connect],
/// and [Graph::disconnect] methods.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct NodeReference(usize);

/// Use this value in the [Graph::connect] method to indicate that a node receives external input.
pub const EXTERNAL_INPUT: NodeReference = NodeReference(0);

/// Orchestrate a group of [processing nodes](Node) and use that network to handle signal data.
///
/// Sample rate and buffer length (in samples) must be constant at compile time, but the nodes and
/// connections can be configured at runtime.
///
/// ### Example
/// ```
/// # use agora::{Buffer, Graph};
/// let mut graph = Graph::<160, 48_000>::default();
/// let mut buffer = [0.0; 160];
/// graph.process(&mut buffer);
/// assert_eq!(buffer, [0.0; 160]);
/// ```
///
/// ### Implementation notes
///
/// The relationships between nodes are represented as a modified adjacency list. The modifications
/// include bidirectional relationship tracking, a cached list of nodes in topological order, and
/// the use of constant-time indexing for better amortized read performance. All of these cost
/// memory, but save processing power when orchestrating nodes for real-time performance.
#[derive(Default)]
pub struct Graph<const BUFFER_SIZE: usize, const SAMPLE_RATE: usize> {
    current_key: usize,
    nodes: HashMap<usize, Box<dyn Node<BUFFER_SIZE, SAMPLE_RATE>>>,
    connections: HashMap<usize, HashSet<usize>>,
    connections_reverse: HashMap<usize, HashSet<usize>>,
    // topological sort, updated when connections are made
    node_order: Vec<usize>,
}

impl<const BUFFER_SIZE: usize, const SAMPLE_RATE: usize> Graph<BUFFER_SIZE, SAMPLE_RATE> {
    /// Inserts a node into the signal graph, returning its ID.
    ///
    /// Note that simply adding a node does not connect it to any other nodes. The key returned
    /// from this method can be passed into the `remove`, `connect`, and `disconnect` methods.
    pub fn add_node(&mut self, node: Box<dyn Node<BUFFER_SIZE, SAMPLE_RATE>>) -> NodeReference {
        // this will skip zero, intentionally. see `EXTERNAL_INPUT` for why.
        self.current_key += 1;
        self.nodes.insert(self.current_key, node);
        self.node_order.push(self.current_key);
        NodeReference(self.current_key)
    }

    /// Removes a node from the signal graph.
    ///
    /// This also removes all connections to and from this node. Returns true if the node was found
    /// and removed.
    pub fn remove_node(&mut self, key: NodeReference) -> bool {
        self.connections.remove(&key.0).map(|dependents| {
            for dependent in dependents {
                self.connections_reverse
                    .entry(dependent)
                    .and_modify(|dependencies| {
                        dependencies.remove(&key.0);
                    });
            }
        });
        self.connections_reverse.remove(&key.0).map(|dependencies| {
            for dependency in dependencies {
                self.connections.entry(dependency).and_modify(|dependents| {
                    dependents.remove(&key.0);
                });
            }
        });

        self.node_order.retain(|&v| v != key.0);

        self.nodes.remove(&key.0).is_some()
    }

    /// Create a connection between a source node and a sink node.
    ///
    /// When signal is generated, the output of the source will be provided to the sink. Pass in a
    /// source of [EXTERNAL_INPUT] when you want the sink node to receive external input.
    ///
    /// ### Errors
    ///
    /// This method will return a [ConnectError] under the following conditions:
    ///
    /// - Attempting to connect a node to itself (`source == sink`)
    /// - Attempting to create a cycle in the node graph
    pub fn connect(
        &mut self,
        source: NodeReference,
        sink: NodeReference,
    ) -> Result<(), ConnectError> {
        // no self connections or cycles allowed
        if source == sink {
            return Err(ConnectError::SelfReference);
        }
        if self.would_have_cycle(source, sink) {
            return Err(ConnectError::CyclicalReference);
        }

        self.connections
            .entry(source.0)
            .or_insert_with(HashSet::new)
            .insert(sink.0);
        self.connections_reverse
            .entry(sink.0)
            .or_insert_with(HashSet::new)
            .insert(source.0);

        self.refresh_node_order();

        Ok(())
    }

    /// Check for a potential cycle in the graph, before creating one.
    fn would_have_cycle(&self, source: NodeReference, sink: NodeReference) -> bool {
        self.connections_reverse
            .get(&source.0)
            .map(|dependencies| {
                // do a depth first search
                dependencies
                    .iter()
                    .any(|&d| d == sink.0 || self.would_have_cycle(NodeReference(d), sink))
            })
            .unwrap_or(false)
    }

    /// Rebuild processing order via topological sort
    fn refresh_node_order(&mut self) {
        self.node_order.clear();
        let mut incoming_edges_copy = self.connections_reverse.clone();
        let mut no_incoming_edges: Vec<_> = self
            .nodes
            .keys()
            .filter(|&k| {
                self.connections_reverse
                    .get(k)
                    .map(HashSet::is_empty)
                    .unwrap_or(true)
            })
            .copied()
            .collect();

        // basically Kahn's algo, https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
        let empty_hash_set = HashSet::new();
        while !no_incoming_edges.is_empty() {
            let current_node = no_incoming_edges.pop().unwrap();
            self.node_order.push(current_node);

            for &destination in self
                .connections
                .get(&current_node)
                .unwrap_or(&empty_hash_set)
            {
                incoming_edges_copy.entry(destination).and_modify(|e| {
                    e.remove(&current_node);
                });
                if incoming_edges_copy
                    .get(&destination)
                    .map(HashSet::is_empty)
                    .unwrap_or(true)
                {
                    no_incoming_edges.push(destination);
                }
            }
        }
    }

    /// Remove a connection between nodes.
    ///
    /// Returns true if a connection was found and removed.
    pub fn disconnect(&mut self, source: NodeReference, sink: NodeReference) -> bool {
        let mut did_disconnect = false;

        self.connections.entry(source.0).and_modify(|e| {
            did_disconnect |= e.remove(&sink.0);
        });
        self.connections_reverse.entry(sink.0).and_modify(|e| {
            did_disconnect |= e.remove(&source.0);
        });

        did_disconnect
    }

    /// Process the given buffer of input signal.
    ///
    /// The `buffer` parameter serves as both input and output.
    pub fn process(&mut self, buffer: &mut Buffer<BUFFER_SIZE>) {
        let mut buffer_cache = HashMap::new();
        buffer_cache.insert(EXTERNAL_INPUT.0, buffer.to_owned());

        let empty_hash_set = HashSet::new();
        for (key, node) in &mut self.nodes {
            let mut buf = [0.0; BUFFER_SIZE];
            sum_buffers(
                self.connections_reverse
                    .get(key)
                    .unwrap_or(&empty_hash_set)
                    .iter()
                    .map(|i| {
                        buffer_cache
                            .get(i)
                            .expect("Cache miss - this indicates a logic error in agora::Graph")
                    }),
                &mut buf,
            );
            node.process(&mut buf);
            buffer_cache.insert(*key, buf);
        }

        let mut terminal_nodes: Vec<_> = self
            .nodes
            .keys()
            .filter(|&k| {
                self.connections
                    .get(k)
                    .map(HashSet::is_empty)
                    .unwrap_or(true)
            })
            .copied()
            .collect();

        if terminal_nodes.is_empty() {
            terminal_nodes = vec![EXTERNAL_INPUT.0];
        }

        *buffer = [0.0; BUFFER_SIZE];
        sum_buffers(
            terminal_nodes.into_iter().map(|key| {
                buffer_cache
                    .get(&key)
                    .expect("Cache miss - this indicates a logic error in agora::Graph")
            }),
            buffer,
        );
    }
}

/// Describes the type of error encountered when [making a connection](Graph::connect) between Nodes.
#[derive(Debug, PartialEq)]
pub enum ConnectError {
    /// Returned when a connection is attempted between a Node and itself.
    SelfReference,
    /// Returned when a connection is attempted that would create a cycle in the graph.
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
            .for_each(|sample| *sample /= num_inputs as f64);
    }
}

#[cfg(test)]
mod node_management {
    use super::{Buffer, ConnectError, Graph, Node, EXTERNAL_INPUT};

    struct NopNode;
    impl<const B: usize, const S: usize> Node<B, S> for NopNode {
        fn process(&mut self, _buffer: &mut Buffer<B>) {}
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
        assert!(!g.remove_node(EXTERNAL_INPUT));
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
    use std::f64::consts::{PI, TAU};

    struct DcOffsetNode(f64);
    impl<const B: usize, const S: usize> Node<B, S> for DcOffsetNode {
        fn process(&mut self, buffer: &mut Buffer<B>) {
            buffer.iter_mut().for_each(|v| *v += self.0);
        }
    }

    #[test]
    fn generates_basic_signal() {
        let mut g = Graph::<20, 1>::default();
        g.add_node(Box::new(DcOffsetNode(1.0)));
        let mut buffer = [0.0; 20];
        g.process(&mut buffer);
        assert_eq!(buffer, [1.0; 20]);
    }

    #[test]
    fn processes_external_signal() {
        let mut g = Graph::<20, 1>::default();
        let node_a = g.add_node(Box::new(DcOffsetNode(-2.0)));
        g.connect(EXTERNAL_INPUT, node_a).unwrap();
        let mut buffer = [1.0; 20];
        g.process(&mut buffer);
        assert_eq!(buffer, [-1.0; 20]);
    }

    struct PulseNode {
        frequency: f64,
        current_phase: f64,
    }
    impl<const B: usize, const S: usize> Node<B, S> for PulseNode {
        fn process(&mut self, buffer: &mut Buffer<B>) {
            buffer.fill_with(|| {
                let last_phase = self.current_phase;
                self.current_phase += TAU / (self.frequency * S as f64);
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
        g.process(&mut buffer);
        let expected = [
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
        ];
        assert_eq!(buffer, expected);
    }
}
