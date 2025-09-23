struct gravity_gpu_values_host{
	/* floats needed for GPU calculations on CPU*/
	float *h_i;
	float *h_j;
	float *mass_i;
	float *mass_j;
	float *x_i;
	float *x_j;
	float *y_i;
	float *y_j;
	float *z_i;
	float *z_j;
	float *a_x_i;
	float *a_y_i;
	float *a_z_i;
	float *a_x_j;
	float *a_y_j;
	float *a_z_j;
	float *pot_i;
	float *pot_j;
	int *active_i;
	int *active_j;
	float *CoM_i;
	float *CoM_j;
	int *gcounts;
	};
	
struct gravity_gpu_values_device{
	/* floats needed for GPU calculations on GPU*/
	float *d_h_i;
	float *d_h_j;
	float *d_mass_i;
	float *d_mass_j;
	float *d_x_i;
	float *d_x_j;
	float *d_y_i;
	float *d_y_j;
	float *d_z_i;
	float *d_z_j;
	float *d_a_x_i;
	float *d_a_y_i;
	float *d_a_z_i;
	float *d_a_x_j;
	float *d_a_y_j;
	float *d_a_z_j;
	float *d_pot_i;
	float *d_pot_j;
	int *d_active_i;
	int *d_active_j;
	float *d_CoM_i;
	float *d_CoM_j;
	int * d_gcounts;
	};
	
void gravity_gpu_allocate_mem_host(struct gravity_gpu_values_host *gravity_gpu_values, int ncells, int max_cell_size){
	//allocate memory on host
	cudaMallocHost((void **)&gravity_gpu_values->h_i, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->h_j, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->mass_i, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->mass_j, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->x_i, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->x_j, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->y_i, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->y_j, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->z_i, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->z_j, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->a_x_i, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->a_y_i, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->a_z_i, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->a_x_j, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->a_y_j, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->a_z_j, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->pot_i, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->pot_j, ncells * max_cell_size * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->active_i, ncells * max_cell_size * sizeof(int));
	cudaMallocHost((void **)&gravity_gpu_values->active_j, ncells * max_cell_size * sizeof(int));
	cudaMallocHost((void **)&gravity_gpu_values->CoM_i, ncells * 3 * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->CoM_j, ncells * 3 * sizeof(float));
	cudaMallocHost((void **)&gravity_gpu_values->gcounts, ncells * sizeof(int));
	}

void gravity_gpu_allocate_mem_device(struct gravity_gpu_values_device *gravity_gpu_values, int ncells, int max_cell_size){
	//allocate memory on device
	cudaMalloc((void **)&gravity_gpu_values->d_h_i, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_h_j, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_mass_i, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_mass_j, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_x_i, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_x_j, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_y_i, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_y_j, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_z_i, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_z_j, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_a_x_i, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_a_y_i, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_a_z_i, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_a_x_j, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_a_y_j, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_a_z_j, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_pot_i, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_pot_j, ncells * max_cell_size * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_active_i, ncells * max_cell_size * sizeof(int));
	cudaMalloc((void **)&gravity_gpu_values->d_active_j, ncells * max_cell_size * sizeof(int));
	cudaMalloc((void **)&gravity_gpu_values->d_CoM_i, ncells * 3 * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_CoM_j, ncells * 3 * sizeof(float));
	cudaMalloc((void **)&gravity_gpu_values->d_gcounts, ncells * sizeof(int));
	}
	
void gravity_gpu_fill_arrays(struct gravity_gpu_values_host *gravity_gpu_values, struct gravity_cache *const ci_cache, int pack_count, int max_cell_size, int gcount){
	//put values into long arrays
  	    for (int i = 0; i < gcount; i++){
            	gravity_gpu_values->h_i[i + pack_count*max_cell_size] = ci_cache->epsilon[i];
            	gravity_gpu_values->h_j[i + pack_count*max_cell_size] = ci_cache->epsilon[i];
            	gravity_gpu_values->mass_i[i + pack_count*max_cell_size] = ci_cache->m[i];
            	gravity_gpu_values->mass_j[i + pack_count*max_cell_size] = ci_cache->m[i];
            	gravity_gpu_values->x_i[i + pack_count*max_cell_size] = ci_cache->x[i];
            	gravity_gpu_values->x_j[i + pack_count*max_cell_size] = ci_cache->x[i];
            	gravity_gpu_values->y_i[i + pack_count*max_cell_size] = ci_cache->y[i];
            	gravity_gpu_values->y_j[i + pack_count*max_cell_size] = ci_cache->y[i];
            	gravity_gpu_values->z_i[i + pack_count*max_cell_size] = ci_cache->z[i];
            	gravity_gpu_values->z_j[i + pack_count*max_cell_size] = ci_cache->z[i];
            	gravity_gpu_values->a_x_i[i + pack_count*max_cell_size] = ci_cache->a_x[i];
            	gravity_gpu_values->a_x_j[i + pack_count*max_cell_size] = ci_cache->a_x[i];
            	gravity_gpu_values->a_y_i[i + pack_count*max_cell_size] = ci_cache->a_y[i];
            	gravity_gpu_values->a_y_j[i + pack_count*max_cell_size] = ci_cache->a_y[i];
            	gravity_gpu_values->a_z_i[i + pack_count*max_cell_size] = ci_cache->a_z[i];
            	gravity_gpu_values->a_z_j[i + pack_count*max_cell_size] = ci_cache->a_z[i];
            	gravity_gpu_values->pot_i[i + pack_count*max_cell_size] = ci_cache->pot[i];
            	gravity_gpu_values->pot_j[i + pack_count*max_cell_size] = ci_cache->pot[i];
            	gravity_gpu_values->active_i[i + pack_count*max_cell_size] = ci_cache->active[i];
            	gravity_gpu_values->active_j[i + pack_count*max_cell_size] = ci_cache->active[i];
            	//add two arrays for each particle to idenify where cj starts and ends
            }
            
            /*for (int i =0; i < 3; i++){
            	CoM_i[pack_count*max_cell_size + i] = ci_cache->x[i];
            	CoM_j[pack_count*max_cell_size + i] = cj_cache->x[i];
            	}*/
            }
            
void gravity_gpu_H2D(struct gravity_gpu_values_host *gravity_gpu_values_h, struct gravity_gpu_values_device *gravity_gpu_values_d, int ncells, int max_cell_size){
	//now copy all the arrays to the device
        cudaMemcpyAsync(gravity_gpu_values_d->d_h_i, gravity_gpu_values_h->h_i, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
        cudaMemcpyAsync(gravity_gpu_values_d->d_h_j, gravity_gpu_values_h->h_j, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_mass_i, gravity_gpu_values_h->mass_i, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_mass_j, gravity_gpu_values_h->mass_j, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_x_i, gravity_gpu_values_h->x_i, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_y_i, gravity_gpu_values_h->y_i, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_z_i, gravity_gpu_values_h->z_i, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_x_j, gravity_gpu_values_h->x_j, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_y_j, gravity_gpu_values_h->y_j, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_z_j, gravity_gpu_values_h->z_j, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_a_x_i, gravity_gpu_values_h->a_x_i, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_a_y_i, gravity_gpu_values_h->a_y_i, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_a_z_i, gravity_gpu_values_h->a_z_i, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_a_x_j, gravity_gpu_values_h->a_x_j, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_a_y_j, gravity_gpu_values_h->a_y_j, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_a_z_j, gravity_gpu_values_h->a_z_j, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_pot_i, gravity_gpu_values_h->pot_i, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_pot_j, gravity_gpu_values_h->pot_j, ncells * max_cell_size * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_active_i, gravity_gpu_values_h->active_i, ncells * max_cell_size * sizeof(int), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_active_j, gravity_gpu_values_h->active_j, ncells * max_cell_size * sizeof(int), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_CoM_i, gravity_gpu_values_h->CoM_i, ncells * 3 * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_CoM_j, gravity_gpu_values_h->CoM_j, ncells * 3 * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(gravity_gpu_values_d->d_gcounts, gravity_gpu_values_h->gcounts, ncells * sizeof(int), cudaMemcpyHostToDevice, 0);
		}
		
void gravity_gpu_D2H(struct gravity_gpu_values_host *gravity_gpu_values_h, struct gravity_gpu_values_device *gravity_gpu_values_d, int ncells, int max_cell_size){
	cudaMemcpyAsync(gravity_gpu_values_h->a_x_i, gravity_gpu_values_d->d_a_x_i, ncells * max_cell_size * sizeof(float), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(gravity_gpu_values_h->a_y_i, gravity_gpu_values_d->d_a_y_i, ncells * max_cell_size * sizeof(float), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(gravity_gpu_values_h->a_z_i, gravity_gpu_values_d->d_a_z_i, ncells * max_cell_size * sizeof(float), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(gravity_gpu_values_h->a_x_j, gravity_gpu_values_d->d_a_x_j, ncells * max_cell_size * sizeof(float), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(gravity_gpu_values_h->a_y_j, gravity_gpu_values_d->d_a_y_j, ncells * max_cell_size * sizeof(float), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(gravity_gpu_values_h->a_z_j, gravity_gpu_values_d->d_a_z_j, ncells * max_cell_size * sizeof(float), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(gravity_gpu_values_h->pot_i, gravity_gpu_values_d->d_pot_i, ncells * max_cell_size * sizeof(float), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(gravity_gpu_values_h->pot_j, gravity_gpu_values_d->d_pot_j, ncells * max_cell_size * sizeof(float), cudaMemcpyDeviceToHost, 0);
	}
