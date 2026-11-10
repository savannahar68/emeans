use std::fmt;

/// Errors returned by training and assignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A setting or an input has the wrong value or shape.
    InvalidInput(String),
    /// A method of the batch loop was called at the wrong time, or with the wrong number of rows.
    WrongOrder(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Error::WrongOrder(msg) => write!(f, "wrong call order: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::InvalidInput(msg.into()))
}

pub(crate) fn wrong_order<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::WrongOrder(msg.into()))
}
