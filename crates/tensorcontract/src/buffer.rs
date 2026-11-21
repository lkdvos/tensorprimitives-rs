//! Over-aligned scratch buffers for packed panels.

use std::alloc::{alloc, dealloc, Layout};

/// Cache-line / AVX-512 friendly alignment for packed panels.
pub(crate) const PANEL_ALIGN: usize = 64;

/// A 64-byte-aligned, uninitialised-on-creation buffer of `T`.
///
/// Packed panels are fully written before being read (the packing routine
/// zero-fills edge rows), so the contents start out as whatever the allocator
/// handed back. `as_mut_ptr` is the only way in.
pub(crate) struct Panel<T> {
    ptr: *mut T,
    layout: Layout,
}

impl<T> Panel<T> {
    pub fn new(len: usize) -> Self {
        let len = len.max(1);
        let layout = Layout::from_size_align(len * core::mem::size_of::<T>(), PANEL_ALIGN)
            .expect("panel layout")
            .pad_to_align();
        // SAFETY: size is non-zero because len >= 1 and T is a float type.
        let ptr = unsafe { alloc(layout) } as *mut T;
        assert!(
            !ptr.is_null(),
            "failed to allocate {} byte panel",
            layout.size()
        );
        Panel { ptr, layout }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T> Drop for Panel<T> {
    fn drop(&mut self) {
        // SAFETY: allocated with exactly this layout in `new`.
        unsafe { dealloc(self.ptr as *mut u8, self.layout) }
    }
}

// SAFETY: `Panel` owns its allocation exclusively and `T` is a plain scalar.
unsafe impl<T: Send> Send for Panel<T> {}

/// A growable 64-byte-aligned buffer, reused across serial contractions on one
/// thread. Contents are never read before being written.
pub(crate) struct Scratch {
    ptr: *mut u8,
    cap: usize,
}

impl Scratch {
    pub const fn empty() -> Self {
        Scratch {
            ptr: core::ptr::null_mut(),
            cap: 0,
        }
    }

    /// Three disjoint 64-byte-aligned regions of `len0`, `len1` and `len2`
    /// elements of `T`.
    pub fn carve3<T>(&mut self, len0: usize, len1: usize, len2: usize) -> [*mut T; 3] {
        let sz = core::mem::size_of::<T>();
        let bytes = |n: usize| (n.max(1) * sz).next_multiple_of(PANEL_ALIGN);
        let (b0, b1, b2) = (bytes(len0), bytes(len1), bytes(len2));
        self.reserve(b0 + b1 + b2);
        // SAFETY: the three regions lie inside the `cap >= b0 + b1 + b2` bytes.
        unsafe {
            [
                self.ptr as *mut T,
                self.ptr.add(b0) as *mut T,
                self.ptr.add(b0 + b1) as *mut T,
            ]
        }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    fn reserve(&mut self, bytes: usize) {
        if bytes <= self.cap {
            return;
        }
        let layout = Layout::from_size_align(bytes, PANEL_ALIGN).expect("scratch layout");
        // SAFETY: `bytes` is non-zero (it exceeds `cap >= 0`), layout is valid.
        let ptr = unsafe { alloc(layout) };
        assert!(
            !ptr.is_null(),
            "failed to allocate {bytes} byte scratch buffer"
        );
        self.release();
        self.ptr = ptr;
        self.cap = bytes;
    }

    fn release(&mut self) {
        if self.cap != 0 {
            // SAFETY: allocated in `reserve` with exactly this layout.
            unsafe {
                dealloc(
                    self.ptr,
                    Layout::from_size_align_unchecked(self.cap, PANEL_ALIGN),
                )
            }
            self.cap = 0;
            self.ptr = core::ptr::null_mut();
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        self.release();
    }
}

// SAFETY: owns its allocation exclusively.
unsafe impl Send for Scratch {}
