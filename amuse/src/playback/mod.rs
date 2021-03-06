// Score:
//   - A list of instruments
//   - A sequence of sections
//     - Dynamic "markings"

// Section:
//   - Time Signature
//   - Number of measures
//   - Key Signature
//   - Individual parts
//     - Instrument
//     - Notes
pub mod choir;
pub mod voice;

pub enum Scale {
    Do,
    Re,
    Mi,
    Fa,
    So,
    La,
    Ti,
}
