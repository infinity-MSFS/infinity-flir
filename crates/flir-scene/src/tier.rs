//! Fallback tiers: fail closed, automatically, and report which inputs were
//! available. The develop picks the highest tier its captured layers support.

/// Which captured layers are present and usable this frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct Inputs {
    pub depth: bool,
    pub depth_convention_known: bool,
    pub gbuffer: bool,
    pub clouds: bool,
    pub colour: bool,
}

/// The fidelity tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// G-buffer + depth + clouds: full material classification and atmosphere.
    Full = 1,
    /// Geometry, sky, atmosphere, altitude, depth-derivative normals, and a
    /// colour-derived material guess.
    DepthOnly = 2,
    /// Weighted-RGB heat proxy with a cooling (blue, not magenta) sky.
    ColourOnly = 3,
}

impl Tier {
    /// Highest tier the inputs support. Returns `None` when not even a colour
    /// proxy is possible (FLIR unavailable; HUD unaffected).
    pub fn select(i: Inputs) -> Option<Tier> {
        if i.depth && i.depth_convention_known && i.gbuffer {
            Some(Tier::Full)
        } else if i.depth && i.depth_convention_known {
            Some(Tier::DepthOnly)
        } else if i.colour {
            Some(Tier::ColourOnly)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_full_when_all_present() {
        let i = Inputs { depth: true, depth_convention_known: true, gbuffer: true, clouds: true, colour: true };
        assert_eq!(Tier::select(i), Some(Tier::Full));
    }

    #[test]
    fn drops_to_depth_without_gbuffer() {
        let i = Inputs { depth: true, depth_convention_known: true, gbuffer: false, clouds: true, colour: true };
        assert_eq!(Tier::select(i), Some(Tier::DepthOnly));
    }

    #[test]
    fn drops_to_colour_without_usable_depth() {
        let i = Inputs { depth: true, depth_convention_known: false, gbuffer: false, clouds: false, colour: true };
        assert_eq!(Tier::select(i), Some(Tier::ColourOnly));
    }

    #[test]
    fn none_when_nothing_usable() {
        assert_eq!(Tier::select(Inputs::default()), None);
    }
}
