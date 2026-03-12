pub trait ProgressLike: Send + Sync {
    fn set_progress(&mut self, p: usize);
}

pub struct ProgressReporter {
    pub rank: usize,
    pub progress: usize,
}

impl ProgressLike for ProgressReporter {
    fn set_progress(&mut self, p: usize) {
        self.progress = p;
    }
}

impl ProgressReporter {
    pub fn new(rank: usize) -> Self {
        Self { rank, progress: 0 }
    }
}
