#[allow(unused_macros)]
macro_rules! info {
    // Name / target / parent.
    (name: $name:expr, target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (name: $name:expr, target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => {};

    // Name / target.
    (name: $name:expr, target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (name: $name:expr, target: $target:expr, $($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, ?$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, %$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, $($arg:tt)+ ) => {};

    // Target / parent.
    (target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)+ ) => {};
    (target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)+ ) => {};
    (target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)+ ) => {};
    (target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => {};

    // Name / parent.
    (name: $name:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (name: $name:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, parent: $parent:expr, $($arg:tt)+ ) => {};

    // Name.
    (name: $name:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (name: $name:expr, $($k:ident).+ $($field:tt)* ) => {};
    (name: $name:expr, ?$($k:ident).+ $($field:tt)* ) => {};
    (name: $name:expr, %$($k:ident).+ $($field:tt)* ) => {};
    (name: $name:expr, $($arg:tt)+ ) => {};

    // Target.
    (target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (target: $target:expr, $($k:ident).+ $($field:tt)* ) => {};
    (target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => {};
    (target: $target:expr, %$($k:ident).+ $($field:tt)* ) => {};
    (target: $target:expr, $($arg:tt)+ ) => {};

    // Parent.
    (parent: $parent:expr, { $($field:tt)+ }, $($arg:tt)+ ) => {};
    (parent: $parent:expr, $($k:ident).+ = $($field:tt)*) => {};
    (parent: $parent:expr, ?$($k:ident).+ = $($field:tt)*) => {};
    (parent: $parent:expr, %$($k:ident).+ = $($field:tt)*) => {};
    (parent: $parent:expr, $($k:ident).+, $($field:tt)*) => {};
    (parent: $parent:expr, ?$($k:ident).+, $($field:tt)*) => {};
    (parent: $parent:expr, %$($k:ident).+, $($field:tt)*) => {};
    (parent: $parent:expr, $($arg:tt)+) => {};

    // ...
    ({ $($field:tt)+ }, $($arg:tt)+ ) => {};
    ($($k:ident).+ = $($field:tt)*) => {};
    (?$($k:ident).+ = $($field:tt)*) => {};
    (%$($k:ident).+ = $($field:tt)*) => {};
    ($($k:ident).+, $($field:tt)*) => {};
    (?$($k:ident).+, $($field:tt)*) => {};
    (%$($k:ident).+, $($field:tt)*) => {};
    (?$($k:ident).+) => {};
    (%$($k:ident).+) => {};
    ($($k:ident).+) => {};
    ($($arg:tt)+) => {};
}

#[allow(unused_macros)]
macro_rules! trace {
    // Name / target / parent.
    (name: $name:expr, target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (name: $name:expr, target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => {};

    // Name / target.
    (name: $name:expr, target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (name: $name:expr, target: $target:expr, $($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, ?$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, %$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, target: $target:expr, $($arg:tt)+ ) => {};

    // Target / parent.
    (target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)+ ) => {};
    (target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)+ ) => {};
    (target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)+ ) => {};
    (target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => {};

    // Name / parent.
    (name: $name:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (name: $name:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)+ ) => {};
    (name: $name:expr, parent: $parent:expr, $($arg:tt)+ ) => {};

    // Name.
    (name: $name:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (name: $name:expr, $($k:ident).+ $($field:tt)* ) => {};
    (name: $name:expr, ?$($k:ident).+ $($field:tt)* ) => {};
    (name: $name:expr, %$($k:ident).+ $($field:tt)* ) => {};
    (name: $name:expr, $($arg:tt)+ ) => {};

    // Target.
    (target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => {};
    (target: $target:expr, $($k:ident).+ $($field:tt)* ) => {};
    (target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => {};
    (target: $target:expr, %$($k:ident).+ $($field:tt)* ) => {};
    (target: $target:expr, $($arg:tt)+ ) => {};

    // Parent.
    (parent: $parent:expr, { $($field:tt)+ }, $($arg:tt)+ ) => {};
    (parent: $parent:expr, $($k:ident).+ = $($field:tt)*) => {};
    (parent: $parent:expr, ?$($k:ident).+ = $($field:tt)*) => {};
    (parent: $parent:expr, %$($k:ident).+ = $($field:tt)*) => {};
    (parent: $parent:expr, $($k:ident).+, $($field:tt)*) => {};
    (parent: $parent:expr, ?$($k:ident).+, $($field:tt)*) => {};
    (parent: $parent:expr, %$($k:ident).+, $($field:tt)*) => {};
    (parent: $parent:expr, $($arg:tt)+) => {};

    // ...
    ({ $($field:tt)+ }, $($arg:tt)+ ) => {};
    ($($k:ident).+ = $($field:tt)*) => {};
    (?$($k:ident).+ = $($field:tt)*) => {};
    (%$($k:ident).+ = $($field:tt)*) => {};
    ($($k:ident).+, $($field:tt)*) => {};
    (?$($k:ident).+, $($field:tt)*) => {};
    (%$($k:ident).+, $($field:tt)*) => {};
    (?$($k:ident).+) => {};
    (%$($k:ident).+) => {};
    ($($k:ident).+) => {};
    ($($arg:tt)+) => {};
}
