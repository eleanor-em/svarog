// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// TODO: macro for N buffers

use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

use vulkano::pipeline::vertex::{Vertex, VertexSource, InputRate, AttributeInfo, IncompatibleVertexDefinitionError, VertexDefinition};
use vulkano::buffer::{TypedBufferAccess, BufferAccess};
use vulkano::pipeline::shader::ShaderInterfaceDef;

/// Unstable.
// TODO: shouldn't be just `Three` but `Multi`
pub struct ThreeBuffersDefinition<T, U, V>(pub PhantomData<(T, U, V)>);

impl<T, U, V> ThreeBuffersDefinition<T, U, V> {
    #[inline]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

unsafe impl<T, U, V, I> VertexDefinition<I> for ThreeBuffersDefinition<T, U, V>
    where
        T: Vertex,
        U: Vertex,
        V: Vertex,
        I: ShaderInterfaceDef,
{
    type BuffersIter = VecIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

    fn definition(
        &self,
        interface: &I,
    ) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        let attrib = {
            let mut attribs = Vec::with_capacity(interface.elements().len());
            for e in interface.elements() {
                let name = e.name.as_ref().unwrap();

                let (infos, buf_offset) = if let Some(infos) = <T as Vertex>::member(name) {
                    (infos, 0)
                } else if let Some(infos) = <U as Vertex>::member(name) {
                    (infos, 1)
                }  else if let Some(infos) = <V as Vertex>::member(name) {
                    (infos, 2)
                } else {
                    return Err(IncompatibleVertexDefinitionError::MissingAttribute {
                        attribute: name.clone().into_owned(),
                    });
                };

                if !infos.ty.matches(
                    infos.array_size,
                    e.format,
                    e.location.end - e.location.start,
                ) {
                    return Err(IncompatibleVertexDefinitionError::FormatMismatch {
                        attribute: name.clone().into_owned(),
                        shader: (e.format, (e.location.end - e.location.start) as usize),
                        definition: (infos.ty, infos.array_size),
                    });
                }

                let mut offset = infos.offset;
                for loc in e.location.clone() {
                    attribs.push((
                        loc,
                        buf_offset,
                        AttributeInfo {
                            offset: offset,
                            format: e.format,
                        },
                    ));
                    offset += e.format.size().unwrap();
                }
            }
            attribs
        }
            .into_iter(); // TODO: meh

        let buffers = vec![
            (0, mem::size_of::<T>(), InputRate::Vertex),
            (1, mem::size_of::<U>(), InputRate::Vertex),
            (2, mem::size_of::<V>(), InputRate::Vertex),
        ]
            .into_iter();

        Ok((buffers, attrib))
    }
}

unsafe impl<T, U, V> VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>>
for ThreeBuffersDefinition<T, U, V>
    where
        T: Vertex,
        U: Vertex,
        V: Vertex,
{
    #[inline]
    fn decode(
        &self,
        source: Vec<Arc<dyn BufferAccess + Send + Sync>>,
    ) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
        // FIXME: safety
        assert_eq!(source.len(), 3);
        let vertices = [
            source[0].size() / mem::size_of::<T>(),
            source[1].size() / mem::size_of::<U>(),
            source[2].size() / mem::size_of::<V>(),
        ]
            .iter()
            .cloned()
            .min()
            .unwrap();
        (
            vec![Box::new(source[0].clone()), Box::new(source[1].clone()), Box::new(source[2].clone())],
            vertices,
            1,
        )
    }
}

unsafe impl<'a, T, U, V, Bt, Bu, Bv> VertexSource<(Bt, Bu, Bv)> for ThreeBuffersDefinition<T, U, V>
    where
        T: Vertex,
        Bt: TypedBufferAccess<Content = [T]> + Send + Sync + 'static,
        U: Vertex,
        Bu: TypedBufferAccess<Content = [U]> + Send + Sync + 'static,
        V: Vertex,
        Bv: TypedBufferAccess<Content = [V]> + Send + Sync + 'static,
{
    #[inline]
    fn decode(&self, source: (Bt, Bu, Bv)) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
        let vertices = [source.0.len(), source.1.len(), source.2.len()]
            .iter()
            .cloned()
            .min()
            .unwrap();
        (
            vec![Box::new(source.0) as Box<_>, Box::new(source.1) as Box<_>, Box::new(source.2) as Box<_>],
            vertices,
            1,
        )
    }
}
