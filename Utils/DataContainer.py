class DataContainer:
    def __init__(self, inputs, outputs, k = 1):
        self.inputs = inputs
        self.outputs = outputs
        self.create_cross_validation_batches(k)

    def _get_batch(self, inputs, outputs, epoch, batch_size):
        start = epoch * batch_size
        end = start + batch_size
        x, y = inputs[start:end], outputs[start:end]
        return x, y

    def next_batch(self, data, epoch, batch_size):
        x, y = self._get_batch(data['inputs'], data['outputs'], epoch, batch_size)
        return x, y

    def create_cross_validation_batches(self, k = 1):
        self.cross_validation_batches = []
        self.current_cv_batch_n = 0

        batch_size = int(len(self.inputs) / k)

        batches = []

        for _ in range(k):
            start   = k * batch_size
            end     = start + batch_size

            batches.append({
                'inputs': self.inputs[start:end],
                'outputs': self.outputs[start:end]
            })

        for valid_idx in range(len(batches)):
            training_data = { 'inputs': [], 'outputs': [] }
            testing_data  = { 'inputs': [], 'outputs': [] }

            for i in range(len(batches)):
                batch = batches[i]
                if i is not valid_idx:
                    training_data['inputs'].extend(batch['inputs'])
                    training_data['outputs'].extend(batch['outputs'])
                else:
                    testing_data = batch

            self.cross_validation_batches.append({
                'train': training_data,
                'test': testing_data
            })

        return self.cross_validation_batches