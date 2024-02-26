class AttackProgress(BaseModel):
    message_type: str = "AttackProgress"
    current_iteration: int = 0
    max_iterations: int = 0
    current_restart: int = 0
    max_restarts: int = 0
    current_batch: int = 0
    max_batches: int = 0