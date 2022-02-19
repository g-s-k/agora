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
//! ## Assumptions and caveats
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
//! - The time complexity of adding a node or removing a connection is constant (amortized).
//! - The time complexity of replacing or removing a node is linear in the number of nodes.
//! - The time complexity of connecting two nodes is proportional to the number of nodes times the
//! number of existing connections. This is due to the recalculation of the critical path through
//! the graph.
//!
//! ## Example
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
//!
//! ## Afterthoughts
//!
//! There are a few design decisions made in this library that may deserve more consideration.
//!
//! ### Use of const generics for buffer size and sample rate
//!
//! While this is very nice in terms of safety (incompatible processing elements are found
//! immediately at compile time), it is not flexible at all to different platforms and
//! environments. There are many differences between different host operating systems in terms
//! of handling audio, and it may be preferable to support some level of configurability if
//! building an application intended to be cross-platform.
//!
//! ### Use of standard library
//!
//! The Rust standard library provides an excellent developer experience - however, there are
//! some platforms which do not support it, or at least the memory allocation mechanisms it
//! uses. Building a `no_std` version of this library would not be possible in its current
//! incarnation, and would require significant changes to the code.
//!
//! ### I/O limitations
//!
//! Nodes are only given one buffer that is shared between input and output. Most platforms
//! support (at least) stereo I/O, and many operations (such as dynamically mixing signals)
//! would be made much easier with more flexibility here. The `cpal` library (used here in the
//! example application) utilizes interleaved audio - maybe this would be the right way to
//! handle it.
//!
//! ### Parallelization
//!
//! This implementation is single-threaded. It would not require _significant_ refactoring to
//! parallelize, but _some_ work would still be necessary.
//!
//! ### Panics
//!
//! The only (anticipated) panics in this library are in [Graph::process], essentially as
//! assertions on the validity of the graph's construction. These could be replaced with returned
//! errors without much refactoring effort, but since they are checking the internal logic of the
//! library it may be best to keep them as is.

#![warn(missing_docs)]

use std::collections::{HashMap, HashSet};

/// A buffer of signal samples.
pub type Buffer<const SIZE: usize> = [f64; SIZE];

/// Performs an isolated processing step in a signal graph.
pub trait Node<const BUFFER_SIZE: usize, const SAMPLE_RATE: usize>: Send {
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
    buffers: HashMap<usize, Buffer<BUFFER_SIZE>>,
    // topological sort, updated when connections are made
    node_order: Vec<usize>,
    // nodes which are summed into the output
    terminal_nodes: Vec<usize>,
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
        self.buffers.insert(self.current_key, [0.0; BUFFER_SIZE]);
        self.terminal_nodes.push(self.current_key);
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
        self.terminal_nodes.retain(|&v| v != key.0);

        self.nodes.remove(&key.0).is_some()
    }

    /// Replace an existing node in the graph, without modifying its connections.
    ///
    /// Returns the old Node. Only returns `None` if the old Node is not present in the Graph.
    pub fn replace_node(
        &mut self,
        key: NodeReference,
        new_node: Box<dyn Node<BUFFER_SIZE, SAMPLE_RATE>>,
    ) -> Option<Box<dyn Node<BUFFER_SIZE, SAMPLE_RATE>>> {
        // failsafe for case where somehow an invalid reference is used
        if !self.node_order.contains(&key.0) {
            self.node_order.push(key.0);
        }
        self.nodes.insert(key.0, new_node)
    }

    /// Create a connection between a source node and a sink node.
    ///
    /// When signal is generated, the output of the source will be provided to the sink. Pass in a
    /// source of [EXTERNAL_INPUT] when you want the sink node to receive external input.
    ///
    /// ### Errors
    ///
    /// This method will return an [Error] if a cycle would be introduced in the graph.
    pub fn connect(&mut self, source: NodeReference, sink: NodeReference) -> Result<(), Error> {
        self.connections.entry(source.0).or_default().insert(sink.0);
        self.connections_reverse
            .entry(sink.0)
            .or_default()
            .insert(source.0);
        self.refresh_terminal_nodes();

        match self.refresh_node_order() {
            Ok(new_order) => {
                self.node_order = new_order;
                Ok(())
            }
            Err(e) => {
                self.disconnect(source, sink);
                Err(e)
            }
        }
    }

    /// Rebuild processing order via topological sort
    ///
    /// Note that this method does not mutate the Graph just yet - this is because an error might
    /// be found during this process.
    fn refresh_node_order(&self) -> Result<Vec<usize>, Error> {
        let mut new_node_order = Vec::new();
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

        no_incoming_edges.push(EXTERNAL_INPUT.0);

        // basically Kahn's algo, https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
        while let Some(current_node) = no_incoming_edges.pop() {
            if current_node != EXTERNAL_INPUT.0 {
                new_node_order.push(current_node);
            }

            if let Some(destinations) = self.connections.get(&current_node) {
                for &destination in destinations {
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

        // check for remaining edges (cycles)
        incoming_edges_copy.retain(|_, sources| !sources.is_empty());
        if !incoming_edges_copy.is_empty() {
            Err(Error::CyclicalReference)
        } else {
            Ok(new_node_order)
        }
    }

    fn refresh_terminal_nodes(&mut self) {
        self.terminal_nodes = self
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

        self.refresh_terminal_nodes();

        did_disconnect
    }

    /// Process the given buffer of input signal.
    ///
    /// The `buffer` parameter serves as both input and output.
    pub fn process(&mut self, buffer: &mut Buffer<BUFFER_SIZE>) {
        if self.terminal_nodes.is_empty() {
            // special case: no nodes feeding into output, just a passthrough
            return;
        }

        for key in &self.node_order {
            // i blame the borrow checker for these semantics
            for sample_index in 0..BUFFER_SIZE {
                self.buffers.get_mut(key).unwrap()[sample_index] = self
                    .connections_reverse
                    .get(key)
                    .map(|deps| {
                        deps.iter()
                            .map(|d| {
                                if *d == EXTERNAL_INPUT.0 {
                                    buffer[sample_index]
                                } else {
                                    self.buffers[d][sample_index]
                                }
                            })
                            .sum()
                    })
                    .unwrap_or_default();
            }

            self
                .nodes
                .get_mut(&key)
                .expect("Tried to access node that does not exist - this indicates a logic error in agora::Graph")
            .process(self.buffers.get_mut(&key).unwrap());
        }

        buffer.fill(0.0);
        for node_buffer in self.terminal_nodes.iter().map(|key| {
            self.buffers
                .get(&key)
                .expect("Cache miss - this indicates a logic error in agora::Graph")
        }) {
            for (sample, dependency_sample) in buffer.iter_mut().zip(node_buffer) {
                *sample += dependency_sample;
            }
        }
    }
}

/// Describes the type of error encountered when [making a connection](Graph::connect) between Nodes.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Returned when a connection is attempted that would create a cycle in the graph.
    CyclicalReference,
}

#[cfg(test)]
mod node_management {
    use super::{Buffer, Error, Graph, Node};

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
        let node_a = g.add_node(Box::new(NopNode));
        assert!(g.remove_node(node_a));
        assert!(!g.remove_node(node_a));
    }

    #[test]
    fn can_replace_node() {
        let mut g = Graph::<1, 1>::default();
        let node_a = g.add_node(Box::new(NopNode));
        assert!(g.replace_node(node_a, Box::new(NopNode)).is_some());
    }

    #[test]
    fn reports_replacement_status() {
        let mut g = Graph::<1, 1>::default();
        let node_a = g.add_node(Box::new(NopNode));
        assert!(g.remove_node(node_a));
        assert!(g.replace_node(node_a, Box::new(NopNode)).is_none());
    }

    #[test]
    fn can_connect_nodes() -> Result<(), Error> {
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
        assert_eq!(connect_result.unwrap_err(), Error::CyclicalReference);
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
        assert_eq!(connect_result.unwrap_err(), Error::CyclicalReference);
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
        assert!(g.connect(EXTERNAL_INPUT, node_a).is_ok());
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
